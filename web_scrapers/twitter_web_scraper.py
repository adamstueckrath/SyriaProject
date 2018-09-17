import twint


c = twint.Config()
c.Geo = "36.2012400,37.1611700,5km"
c.Since = "2017-1-1"
c.Lang = "en"
# Custom output format
c.Store_csv = True
c.Custom_csv = ["id", "date", "user_id", "username", "tweet", "hashtags", "location"]
c.Output = "twitter.csv"

twint.run.Search(c)


if __name__ == '__main__':
	main()