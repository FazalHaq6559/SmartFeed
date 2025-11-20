# 📰 SmartFeed — Personalized News Aggregator

SmartFeed is an **AI-driven personalized news platform** that delivers relevant news content based on user interests, browsing history, and reading behavior.  
It combines **Neural Collaborative Filtering (NCF)** with **Content-Based Filtering** to create a **hybrid recommendation system** that adapts to each user.

---

##  Table of Contents
1. [Overview](#-overview)
2. [Objectives](#-objectives)
3. [Features](#-features)
4. [System Architecture](#-system-architecture)
5. [Tech Stack](#-tech-stack)
6. [Model Details](#-model-details)
7. [Demo Video](#-demo-video)
8. [Results & Impact](#-results--impact)
9. [How to Run Locally](#-how-to-run-locally)
10. [Future Improvements](#-future-improvements)
11. [Author](#-author)
12. [License](#-license)

---

##  Overview
**SmartFeed** was developed as part of a final-year research project to tackle **information overload** in online news platforms.  
The system intelligently recommends news by combining user interaction data with article content similarity, ensuring that every user sees what matters most to them.

This project integrates **Machine Learning**, **Natural Language Processing**, and **Full-Stack Web Development**, bridging AI and software engineering principles.

---

##  Objectives
- To develop an intelligent **personalized news aggregator**.
- To implement a **hybrid recommendation model** (NCF + content-based).
- To provide an engaging and **user-centric interface** for browsing news.
- To evaluate the system’s performance with **real-world user data** (10,000+ users).

---

##  Features
-  Secure **user authentication** with JWT.
-  **Hybrid recommendation engine** combining collaborative and content-based signals.
-  **Personalized "For You" section** for individual news suggestions.
-  **FastAPI integration** with Node.js backend.
-  **MongoDB** for user profiles, preferences, and reading history.
-  **React-based frontend** for an interactive experience.
-  Real-time model inference with BERT-based embeddings.

---

##  System Architecture
```
Frontend (React)
     ↓
Backend (Node.js + Express)
     ↓
Model Service (FastAPI + Hybrid Model)
     ↓
Database (MongoDB)
```

- **React.js**: User interface for reading and interacting with news.  
- **Express.js + Node.js**: Handles authentication, routes, and communication with the model service.  
- **FastAPI (Python)**: Runs the hybrid recommendation model using pre-trained embeddings and TensorFlow.  
- **MongoDB**: Stores user information, preferences, and article metadata.

---

##  Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | React.js |
| Backend | Node.js, Express |
| Model Service | FastAPI (Python) |
| Machine Learning | TensorFlow, BERT |
| Database | MongoDB |
| Deployment | Docker / Render / Vercel |
| Authentication | JWT, bcrypt |

---

##  Model Details
- **Dataset:** MIND (Microsoft News Dataset)
- **Embedding Generation:** BERT for news titles and abstracts
- **Hybrid Model:**
  - **Neural Collaborative Filtering (NCF)** for user–news interaction learning
  - **Content-Based Filtering** using cosine similarity on news embeddings
- **Output:** Ranked list of top-N personalized news articles

---



---

##  Demo Video
[▶ Watch Demo Video on YouTube](https://youtu.be/2tIs-oUwR0M)


---

## Results & Impact
- Achieved a **notable increase in engagement** through relevant recommendations.
- Reduced content redundancy using a **hybrid ranking approach**.
- Demonstrated the application of **deep learning + web engineering** for real-world personalization.

---

##  How to Run Locally

### 1. Clone Repository
```bash
git clone https://github.com/YourUsername/SmartFeed.git
cd SmartFeed
```

### 2. Run Node.js Backend
```bash
cd backend/server
npm install
npm start
```

### 3. Run FastAPI Service
```bash
cd ../model_service
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 4. Run Frontend
```bash
cd ../../frontend
npm install
npm start
```

---

##  Future Improvements
-  Develop a **mobile version** for Android/iOS.
-  Add **multilingual news support** (English, Urdu, French, etc.).
-  Implement **explainable AI** for transparency in recommendations.
-  Introduce **real-time trend detection** and topic clustering.
-  Optimize model serving for large-scale deployment.

---

##  Author
**Fazal Haq**  
Bachelor of Computer Science  
Bahria University, Islamabad  
Email: [fhaq6559@gmail.com](mailto:fhaq6559@gmail.com)  

---

## License
This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with proper attribution.

---

> *“SmartFeed bridges artificial intelligence and human curiosity — delivering the right story to the right reader at the right time.”*

