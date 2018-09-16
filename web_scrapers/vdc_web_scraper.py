import bs4
import requests
import pandas
from urllib import parse


def extract_database_name(url):
    parse_url = parse.urlsplit(url).path
    path_url = parse_url.split('/')
    return path_url[3]

def add_url_page_number(url, page_number):
    parse_url = parse.urlsplit(url)
    
    # split url path
    path_url = parse_url.path.split('/')

    # remove and replace page number with page_number
    previous_number = path_url[4]
    path_url.remove(previous_number)
    path_url.insert(4, str(page_number))
    
    # join new_path and add it to the base url
    new_path = '/'.join( t for t in path_url)
    return parse.urljoin(url, new_path)

def vcd_crawl(url_path, total_pages):
    df_list = []
    for page_number in range(1, total_pages+1):
        
        # get url with correct page number
        url = add_url_page_number(url_path, page_number)
        page = requests.get(url)
        
        # logging new page number
        print("getting page {}".format(str(page_number)))
              
        # create a BeautifulSoup object
        soup = bs4.BeautifulSoup(page.text, "lxml")

        # find the table in the html with the class peopleListing
        table = soup.find("table", class_="peopleListing")

        # html to pandas dataframe
        df = pandas.read_html(str(table), header=0, index_col=None)
        
        df_list.append(df[0])
    
    final_df = pandas.concat(df_list)
    final_df.reset_index(drop=True, inplace=True)
    return final_df

def run_crawl(url_paths_pages):
    for url_path, total_pages in url_paths.items():
        dataset = vcd_crawl(url_path, total_pages)
        database_name=extract_database_name(url_path)
        print("writing {database_name}".format(database_name=database_name))
        dataset.to_csv('~/Desktop/vdc_causality_{database_name}.csv'.format(database_name=database_name))

if __name__ == '__main__':

	url_paths = {
			"http://www.vdc-sy.info/index.php/en/martyrs/1/c29ydGJ5PWEua2lsbGVkX2RhdGV8c29ydGRpcj1ERVNDfGFwcHJvdmVkPXZpc2libGV8ZXh0cmFkaXNwbGF5PTB8":1645, 
			"http://www.vdc-sy.info/index.php/en/detainees/1/c29ydGJ5PWEuaTN0ZXFhbF9kYXRlfHNvcnRkaXI9REVTQ3xhcHByb3ZlZD12aXNpYmxlfGV4dHJhZGlzcGxheT0wfA==":667,
			"http://www.vdc-sy.info/index.php/en/missing/1/c29ydGJ5PWEuZGlzYXBlYXJlZF9kYXRlfHNvcnRkaXI9REVTQ3xhcHByb3ZlZD12aXNpYmxlfGV4dHJhZGlzcGxheT0wfA==":30
			}	
	run_crawl(url_paths)
