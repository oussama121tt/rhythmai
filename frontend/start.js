#!/usr/bin/env node
const { execSync } = require('child_process');
const path = require('path');
const process = require('process');

process.chdir(path.dirname(__filename));
execSync('npm run dev', { stdio: 'inherit' });
