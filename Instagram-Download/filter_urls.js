const fs = require('fs')
const ids = new Set(fs.readFileSync('insta_ids.txt').toString().split('\n'))

fs.createReadStream('urls.txt')
.pipe(require('split2')())
.pipe(require('through2')(
	function(chunk, enc, callback) {
		if(ids.has(chunk.toString().split(' ',2)[0])) {
			this.push(chunk)
			this.push('\n')
		}
		callback()
	}
))
.pipe(fs.createWriteStream('filtered_urls.txt'))