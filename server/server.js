const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const axios = require("axios");
const { body, validationResult } = require("express-validator");
const cors = require("cors");
require("dotenv").config();

const app = express();
//app.use(errorHandler);

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
// app.use(cors());

app.use(
  cors({
    origin: ['http://localhost:3002', 'http://127.0.0.1:3002'],
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
    credentials: true,
  })
);

// Connect to MongoDB
mongoose
  .connect(process.env.MONGODB_URI)
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.error("MongoDB connection error:", err));

// User Schema
const UserSchema = new mongoose.Schema(
  {
    username: { type: String, required: true, unique: true },
    email: { type: String, required: true, unique: true, lowercase: true },
    password: { type: String, required: true },
    preferences: { type: [String], default: [] },
    history: { type: [String], default: [] },
  },
  { timestamps: true }
);

const User = mongoose.model("User", UserSchema);

// News Schema
const newsSchema = new mongoose.Schema(
  {
    newsId: { type: String, required: true, unique: true },
    title: { type: String, required: true },
    description: { type: String },
    url: { type: String, required: true },
    category: { type: String, default: "general" },
    country: { type: String, default: "unknown" },
    publishedAt: { type: Date, required: true },
    source: { type: String, default: "unknown" },
    content: { type: String, default: "" },
    urlToImage: { type: String },
    author: { type: String, default: "unknown" },
    embeddings: { type: Array, default: [] },
  },
  {
    timestamps: true, // Add timestamps for tracking when articles are added/updated
  }
);

const News = mongoose.model("News", newsSchema);

// Middleware for JWT authentication
// It checks for a valid JWT token before allowing a request to proceed to protected routes. If the token is missing or invalid, it stops the request and sends an error.
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];
  if (!token) return res.status(401).json({ error: "Access token required" });

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ error: "Invalid or expired token" });
    req.user = user;
    next();
  });
};

// Utility function for making API requests
const makeApiRequest = async (url) => {
  try {
    const response = await axios.get(url);
    return {
      status: 200,
      success: true,
      message: "Successfully fetched the data",
      data: response.data,
    };
  } catch (error) {
    console.error(
      "API request error:",
      error.response ? error.response.data : error
    );
    return {
      status: 500,
      success: false,
      message: "Failed to fetch data from the API",
      error: error.response ? error.response.data : error.message,
    };
  }
};

// Validation middleware
const validateSignup = [
  body("username").isString().notEmpty().withMessage("Username is required"),
  body("email").isEmail().withMessage("Valid email is required"),
  body("password")
    .isLength({ min: 6 })
    .withMessage("Password must be at least 6 characters long"),
];

// Error handling middleware
const errorHandler = (err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: "An unexpected error occurred" });
};

// Signup endpoint
app.post("/signup", validateSignup, async (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }

  try {
    const { username, email, password } = req.body;
    const existingUser = await User.findOne({ $or: [{ username }, { email }] });
    if (existingUser) {
      return res
        .status(400)
        .json({ error: "Username or email already exists" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const user = new User({ username, email, password: hashedPassword });

    await user.save();

    console.log("User created with ID:", user._id.toString());

    try {
      const response = await axios.post(
        `${process.env.FASTAPI_URL}/update_user_mapping`,
        { user_id: user._id.toString() },
        {
          headers: { "Content-Type": "application/json" },
          validateStatus: function (status) {
            return status < 500;
          },
        }
      );
      console.log("FastAPI response:", response.status, response.data);
      if (response.status !== 200) {
        console.error("Failed to update user mapping:", response.data);
      }
    } catch (error) {
      console.error("Error calling FastAPI:", error.message);
      if (error.response) {
        console.error(
          "FastAPI error response:",
          error.response.status,
          error.response.data
        );
      }
    }

    res
      .status(201)
      .json({ message: "User created successfully", userId: user._id });
  } catch (error) {
    next(error);
  }
});

// Login endpoint
app.post("/login", async (req, res, next) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email });
    if (!user) return res.status(400).json({ error: "User not found" });

    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword)
      return res.status(400).json({ error: "Invalid password" });

    const token = jwt.sign({ _id: user._id }, process.env.JWT_SECRET, {
      expiresIn: "1d",
    });
    res.json({ token });
  } catch (error) {
    next(error);
  }
});

// Set initial preferences endpoint
app.post(
  "/setInitialPreferences",
  authenticateToken,
  async (req, res, next) => {
    try {
      const { preferences } = req.body;
      if (!Array.isArray(preferences)) {
        return res.status(400).json({ error: "Preferences must be an array" });
      }

      const user = await User.findById(req.user._id);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      if (user.preferences.length > 0) {
        return res.status(400).json({
          error:
            "Preferences have already been set. Use updatePreferences to modify.",
        });
      }

      user.preferences = preferences;
      await user.save();

      res.json({ message: "Initial preferences set successfully" });
    } catch (error) {
      next(error);
    }
  }
);

// Update preferences endpoint
app.post("/updatePreferences", authenticateToken, async (req, res, next) => {
  try {
    const { preferences } = req.body;
    if (!Array.isArray(preferences)) {
      return res.status(400).json({ error: "Preferences must be an array" });
    }

    const user = await User.findById(req.user._id);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    user.preferences = preferences;
    await user.save();

    res.json({ message: "Preferences updated successfully" });
  } catch (error) {
    next(error);
  }
});

// Update history endpoint
app.post("/updateHistory", authenticateToken, async (req, res, next) => {
  try {
    console.log("Received data:", req.body);
    const { articleId } = req.body;
    if (!articleId) {
      return res.status(400).json({ error: "Article ID is required" });
    }
    await User.findByIdAndUpdate(req.user._id, {
      $push: { history: articleId },
    });
    res.json({ message: "History updated successfully" });
  } catch (error) {
    next(error);
  }
});

// Recommendations endpoint
app.get('/recommendations', authenticateToken, async (req, res, next) => {
  console.log("Reached to backend end point");
  try {
    console.log('Fetching recommendations for user:', req.user._id);
    console.log('FASTAPI_URL:', process.env.FASTAPI_URL);

    const user = await User.findById(req.user._id);
    if (!user) {
      console.log('User not found:', req.user._id);
      return res.status(404).json({ error: 'User not found' });
    }
    console.log('User found:', user._id);
    console.log('User preferences:', user.preferences);
    console.log('User history:', user.history);

    const fastApiUrl = `${process.env.FASTAPI_URL}/recommend/hybrid`;
    console.log('Making request to FastAPI URL:', fastApiUrl);

    const requestData = {
      user_id: user._id.toString(),
      preferences: user.preferences,
      history: user.history
    };
    console.log('Request data:', JSON.stringify(requestData, null, 2));

    const response = await axios.post(fastApiUrl, requestData, {
      headers: {
        'Content-Type': 'application/json'
      }
    });

    console.log('Received response from FastAPI');
    res.json(response.data);
  } catch (error) {
    console.error('Error in recommendations endpoint:', error);
    
    if (error.response) {
      console.error('FastAPI response error:', error.response.status, error.response.data);
      res.status(error.response.status).json({ error: error.response.data });
    } else if (error.request) {
      console.error('No response received from FastAPI:', error.request);
      res.status(500).json({ error: 'No response received from recommendation service' });
    } else {
      console.error('Error setting up request:', error.message);
      res.status(500).json({ error: 'Error setting up request to recommendation service' });
    }
  }
});

// Fetch and store news endpoint
app.get("/fetchNews", authenticateToken, async (req, res, next) => {
  const { pageSize = 80, page = 1, q = "world" } = req.query;
  try {
    const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(
      q
    )}&page=${page}&pageSize=${pageSize}&apiKey=${process.env.API_KEY}`;
    const result = await makeApiRequest(url);

    if (!result.success) {
      return res.status(result.status).json(result);
    }

    // Store articles in MongoDB
    const articles = result.data.articles;

    try {
      // Store articles using Promise.all
      await Promise.all(
        articles.map((article) =>
          News.findOneAndUpdate(
            { newsId: article.url },
            {
              newsId: article.url,
              title: article.title,
              description: article.description || "",
              url: article.url,
              category: q, // Using search query as category
              publishedAt: new Date(article.publishedAt),
              source: article.source.name || "unknown",
              content: article.content || "",
              urlToImage: article.urlToImage || "",
              author: article.author || "unknown",
            },
            { upsert: true }
          )
        )
      );

      console.log(`Successfully stored/updated ${articles.length} articles`);

      // Return the API response after DB operation is complete
      res.status(200).json({
        success: true,
        data: result.data,
        message: `${articles.length} articles stored/updated in database`,
      });
    } catch (dbError) {
      console.error("Error storing articles:", dbError);
      // Still return the API data even if DB storage failed
      res.status(200).json({
        success: true,
        data: result.data,
        warning: "Articles fetched but failed to store in database",
      });
    }
  } catch (error) {
    if (error.response && error.response.status === 429) {
      res.status(429).json({
        success: false,
        message: "Rate limit exceeded. Please try again later.",
      });
    } else {
      next(error);
    }
  }
});

//All news endpoint
app.get("/all-news", async (req, res) => {
  let pageSize = parseInt(req.query.pageSize) || 80;
  let page = parseInt(req.query.page) || 1;
  let q = req.query.q || "world";

  let url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(
    q
  )}&page=${page}&pageSize=${pageSize}&apiKey=${process.env.API_KEY}`;
  const result = await makeApiRequest(url);
  res.status(result.status).json({
    success: true,
    data: result.data,
  });
});

// Top headlines endpoint
app.get("/top-headlines", authenticateToken, async (req, res) => {
  let pageSize = parseInt(req.query.pageSize) || 80;
  let page = parseInt(req.query.page) || 1;
  let category = req.query.category || "general";

  let url = `https://newsapi.org/v2/top-headlines?category=${category}&language=en&page=${page}&pageSize=${pageSize}&apiKey=${process.env.API_KEY}`;
  const result = await makeApiRequest(url);
  res.status(result.status).json(result);
});

// Use error handling middleware
app.use(errorHandler);

const PORT = process.env.PORT;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
