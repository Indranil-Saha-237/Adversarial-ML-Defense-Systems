require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const session = require('express-session');
const MongoStore = require('connect-mongo').default;
const flash = require('express-flash');
const passport = require('./config/passport');
const authRoutes = require('./routes/auth');

const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

// ──── Health check (for Docker / Jenkins) ────
app.get('/health', (req, res) => res.status(200).json({ status: 'ok' }));

// ──── View engine ────
app.set('view engine', 'ejs');
app.set('views', __dirname + '/views');

// ──── Middleware ────
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// ──── Session ────
app.use(session({
  secret: process.env.SESSION_SECRET || 'fallback-secret',
  resave: false,
  saveUninitialized: false,
  store: MongoStore.create({
    mongoUrl: process.env.MONGODB_URI || 'mongodb://localhost:27017/loginpage',
    collectionName: 'sessions'
  }),
  cookie: { maxAge: 1000 * 60 * 60 * 24 } // 1 day
}));

// ──── Passport ────
app.use(passport.initialize());
app.use(passport.session());

// ──── Flash messages ────
app.use(flash());

// ──── Routes ────
app.use('/', authRoutes);

// ──── Connect to MongoDB & start server ────

app.listen(PORT, '0.0.0.0',() => {
    console.log(`Server running on http://localhost:${PORT}`);
    });
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/loginpage')
  .then(()=>{
    console.log('MongoDB connected');
  })
  .catch(err => {
    console.error('MongoDB connection error:', err.message);
    
  });

module.exports = app;

