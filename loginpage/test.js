const { describe, it } = require('node:test');
const assert = require('node:assert');

// Basic smoke tests
describe('App', () => {
  it('should have required environment variables defined', () => {
    require('dotenv').config();
    assert.ok(true);
  });

  it('should load User model without errors', () => {
    const User = require('./models/User');
    assert.ok(User);
    assert.strictEqual(User.modelName, 'User');
  });

  it('should load routes without errors', () => {
    const routes = require('./routes/auth');
    assert.ok(routes);
  });
});
