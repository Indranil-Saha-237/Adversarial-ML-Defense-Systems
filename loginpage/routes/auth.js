const express = require('express');
const router = express.Router();
const passport = require('passport');
const User = require('../models/User');
const axios = require('axios');
 const ML_URL = process.env.ML_API_URL || 'http://127.0.0.1:5001';

// ──── Middleware ────
function ensureGuest(req, res, next) {
  if (req.isAuthenticated()) return res.redirect('/dashboard');
  next();
}

function ensureAuth(req, res, next) {
  if (req.isAuthenticated()) return next();
  res.redirect('/login');
}

// ──── Pages ────
router.get('/', (req, res) => res.redirect('/login'));

router.get('/login', ensureGuest, (req, res) => {
  res.render('login', { error: req.flash('error') });
});

router.get('/signup', ensureGuest, (req, res) => {
  res.render('signup', { error: req.flash('error') });
});

router.get('/dashboard', ensureAuth, async (req, res) => {
  try {
    const mlRes = await axios.get(`${ML_URL}/model/metadata`);
    const mlData = mlRes.data;

    //Reder Dashboard with ML metadata and User Data
    return res.render('dashboard', { user: req.user, ml: mlData });
    
  }
    catch (error) {
    console.error("ML API Error:", error.message);
    }


    return res.render('dashboard', { user: req.user, ml: {
      modelName: "SYSTEM OFFLINE",
      accuracy: 0,
      features_count: 0,
      status : "OFFLINE"
    } });
});

// ──── Local Auth ────
router.post('/login', ensureGuest, passport.authenticate('local', {
  successRedirect: '/dashboard',
  failureRedirect: '/login',
  failureFlash: true
}));

router.post('/signup', ensureGuest, async (req, res) => {
  try {
    const { name, email, password, confirmPassword } = req.body;

    if (!name || !email || !password) {
      req.flash('error', 'Please fill in all fields.');
      return res.redirect('/signup');
    }
    if (password.length < 6) {
      req.flash('error', 'Password must be at least 6 characters.');
      return res.redirect('/signup');
    }
    if (password !== confirmPassword) {
      req.flash('error', 'Passwords do not match.');
      return res.redirect('/signup');
    }

    const existing = await User.findOne({ email: email.toLowerCase() });
    if (existing) {
      req.flash('error', 'Email already registered.');
      return res.redirect('/signup');
    }

    await User.create({ name, email, password });
    req.flash('error', 'Account created! Please log in.');
    res.redirect('/login');
  } catch (err) {
    console.error(err);
    req.flash('error', 'Something went wrong.');
    res.redirect('/signup');
  }
});

// ──── Google Auth ────
router.get('/auth/google',
  passport.authenticate('google', { scope: ['profile', 'email'] })
);

router.get('/auth/google/callback',
  passport.authenticate('google', {
    successRedirect: '/dashboard',
    failureRedirect: '/login',
    failureFlash: true
  })
);

// ──── Logout ────
router.get('/logout', (req, res, next) => {
  req.logout(err => {
    if (err) return next(err);
    res.redirect('/login');
  });
});

// ──── ML Proxy Bridges (For ngrok visibility) ────

// Proxy for Chart Data
router.get('/api/ml/comparison', async (req, res) => {
    try {
        const response = await
axios.get('http://ml-engine:5001/model/comparison');
        res.json(response.data);
    } catch (err) {
       console.error('ML Bridge Error (Comparison):', err.message);
        res.status(503).json({ error: 'ML Engine Offline' });
    }
});
// Proxy for Attack Simulation
router.post('/api/ml/attack', async (req, res) => {
    try {
        const response = await
axios.post('http://ml-engine:5001/attack/simulate', {}, { timeout: 60000 });
        res.json(response.data);
    } catch (err) {
        console.error('ML Bridge Error (Attack):', err.message);
        res.status(503).json({ error: 'Attack Engine Offline' });
    }
});


module.exports = router;
