// scheduler.js
const cron = require('node-cron');
const { fetchAndStoreNews } = require('./controllers/newsController'); // Adjust the path as needed

// Schedule the task to run every hour
cron.schedule('0 * * * *', async () => {
  console.log('Scheduled task: Fetching and storing news...');
  try {
    // Simulate a request object
    const req = {
      query: {
        pageSize: 80,
        page: 1,
        q: 'world' // You can adjust the query as needed
      }
    };

    // Simulate a response object
    const res = {
      status: (statusCode) => ({
        json: (data) => {
          console.log(`Status: ${statusCode}`, data.message);
        }
      })
    };

    await fetchAndStoreNews(req, res);
  } catch (error) {
    console.error('Error during scheduled news fetch:', error);
  }
});

console.log('News fetch scheduler initialized.');
