import twint

def main():
	
	c = twint.Config()
	
	# coordinates of Aleppo, Syria with a 4km radius 
	c.Geo = "36.2012400, 37.1611700, 4km"
	
	# get all tweets from august to septmeber at the locaiton
	c.Since = "2018-8-1"
	c.Until = "2018-9-1"

	# translate arabic to english
	c.Lang = "en"

	# custom output format
	c.Store_csv = True

	# custom twitter fields to capture
	c.Custom_csv = ["id", "date", "user_id", "username", "tweet", "hashtags", "location"]
	c.Output = "twitter.csv"

	# write to file
	twint.run.Search(c)


if __name__ == '__main__':
	main()