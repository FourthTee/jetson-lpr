from openalpr import Alpr
import time
alpr = Alpr("eu", "/etc/openalpr/openalpr.conf","/home/fourth/Desktop/repo/openalpr/runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    

alpr.set_top_n(20)
alpr.set_default_region("md")
start = time.time()
results = alpr.recognize_file("input/exple1.jpg")
print(time.time()-start)
print("Plate: "+results['results'][0]['plate'])
print("Confidence: "+ str(results['results'][0]['confidence']))