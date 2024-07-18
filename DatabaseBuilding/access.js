const sqlite = require('sqlite');
const sqlite3 = require('sqlite3');

async function getDBConnection() {
    const db = await sqlite.open({
        filename:'nutrition.db',
        driver: sqlite3.Database
    });
    return db
}

async function insertFromFile(filename) {
    try{
        let db = await getDBConnection();
        
    }
}