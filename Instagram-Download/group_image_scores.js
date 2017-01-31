const fs = require('fs')

const groups = [
	{
		// needs to be before spring, as May is also included in spring (and should be ignored)
		name: "summer",
		val: require('./locations_summer_may_small.json'),
	},
	{
		name: "spring",
		val: require('./locations_spring_small.json'),
	},
	{
		name: "sept_oct",
		val: require('./locations_sep_oct.json'),
	},
]
const scores = require('./all_urls_sentiment_float.json')
const positions = require('./location_positions.json')
const values = {} // group name => loc id => { scoreSum, count }

let unknownCount = 0

Object.keys(scores).forEach(id => {
	const group = groups.find(g => id in g.val)

	if (!group) {
		unknownCount++
		return
	}
	
	if (!values[group.name]) {
		values[group.name] = {}
	}

	if(!values[group.name][group.val[id].loc_id]) {
		values[group.name][group.val[id].loc_id] = {scoreSum: 0, count: 0}
	}

	values[group.name][group.val[id].loc_id].scoreSum += scores[id]
	values[group.name][group.val[id].loc_id].count++
})

console.log('NOT FOUND: ', unknownCount)

Object.keys(values).forEach(key => {
	const obj = values[key]

	Object.keys(obj).forEach(k => obj[k].score = obj[k].scoreSum / obj[k].count)
	Object.keys(obj).forEach(k => delete obj[k].scoreSum)
	Object.keys(obj).forEach(k => Object.assign(obj[k], positions[k]))

	fs.writeFileSync(`results_${1}.json`, JSON.stringify(Object.keys(obj).map(k => obj[k])))
})