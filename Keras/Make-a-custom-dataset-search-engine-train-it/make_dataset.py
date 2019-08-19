# https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/
# python .\make_dataset.py -q apple -o  images  


from requests import  exceptions
import argparse
import requests
import cv2
import os



ap=argparse.ArgumentParser()
ap.add_argument("-q","--query", required=True,help="Search query to search Bing Image API for")
ap.add_argument('-o','--output',required=True,help="path to output directory of image")

args= vars(ap.parse_args())



API_KEY="079e0b0b670d477f926a21887712c845"
MAX_RESULTS = 250
GROUP_SIZE= 50

URL="https://api.cognitive.microsoft.com/bing/v7.0/images/search"

EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

term=args["query"]
headers={"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# print(os.path.exists(os.path.join(os.getcwd(),args['output'])))
# print(os.path.join(os.getcwd(),args['output']))

if os.path.exists(os.path.join(os.getcwd(),args['output'])):
    pass
else:
    os.system('mkdir -p '+ args['output'])
    print('...[Command][mkdir]'+ 'The {} directory is created.'.format(args['output']) )
    

# make the search
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults,
	term))

# initialize the total number of images downloaded thus far



# loop over the estimated number of results in `GROUP_SIZE` groups
for offset in range(0, estNumResults, GROUP_SIZE):
    print("[INFO] making request for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))
    total = 0
    for v in results["value"]:
        try:
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)
            ext = '.'+v["encodingFormat"]
            p = os.path.sep.join([args["output"], "{}{}".format(
                str(total+offset).zfill(8), ext)])
            total=total+1
            # print(v)
            f = open(p, "wb")
            f.write(r.content)
            f.close()
        except EXCEPTIONS as e:
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue
        image = cv2.imread(p)
        if image is None:
            print("[Warning] deleting:" ,"{}{}".format(str(total+offset).zfill(8), ext))
            os.remove(p)
            continue
            
            
            
