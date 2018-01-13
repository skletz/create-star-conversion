# Phaser GameJam Template

Startup Project for GameJams using Phaser, a HTML 5 media framework based on JavaScript, and Typescript, a free open source programming language developed and maintained by Microsoft. Typescript is a superset of JavaScript and aims at making JavaScript programming more object-oriented. 

## Instructions

### First you should clone this repository 

1. `git clone https://github.com/amplejoe/PhaserGameJamTemplate.git`
2. Optionally remove `.git` folder and rename the project folder to your liking.

### Option A 

Use IntelliJ IDEA Ultimate/Webstorm (free fro students: [Sign Up](https://www.jetbrains.com/student/)). How to import the project:

1. Open IDE, File -> New Project from Existing Sources -> choose previously cloned project folder
2. On 'Import Project' Dialog choose 'Create project from existing sources' -> Next  x 3 -> Finish
 
Compiling with these IDEs is easy, since a Typescript support can be activated in the Settings: 

1. Choose File -> Settings -> Languages & Frameworks -> Typescript
2. Tick Enable TypeScript Compiler and choose 'use tsconfig.json' -> Apply -> OK

### Option B

The template really can be imported into any IDE, just make sure you have installed a TypeScript Compiler. How to:

1. Install [node.js](https://nodejs.org/en/)
2. Install TypeScript compiler from the console/terminal: `npm install -g typescript`  
3. The project can be compiled using `compile_ts.cmd` or `compile_ts.sh`.
4. Import the project into your favourite IDE (and adjust `.gitignore` to ignore IDE created files)

