import subprocess
import sys

inbound = 2
outbound = 3

if len(sys.argv) < 4:
    print ("Usage:  python sentiment_pipeline.py 2 10 <lang> <subpath>")
    sys.exit()

if sys.argv[1]:
    inbound = int(sys.argv[1])

if sys.argv[2]:
    outbound = int(sys.argv[2])

if sys.argv[3]:
    lang = sys.argv[3]

if sys.argv[4]:
    subpath = sys.argv[4]

DataPah = 'data//Twitter_' + lang + '//'
NEResultPath = 'experiments//sentiment//model//baseline_pipeline//Twitter_' + lang + '//' + subpath + '//'
TempPath = 'experiments//sentiment//model//baseline_pipeline//Twitter_' + lang + '//temp//'

print("Start stats: [", inbound, ",",outbound, "] lang=", lang, ' subpath=',subpath)


for i in range(inbound,outbound+1):
    print("2nd Stage of Pipeline: Parse Train/Test %d" % i)

    #subprocess.call("python opendomain_pipeline_createStdNE.py Preprocess\\test.%d.conll.train_test  > pipeline\\result.%d.NE.out" % (i,i), shell=True)

    subprocess.call("python scripts//opendomain_pipeline_preSent.py "  + DataPah + "train.%d.coll t > " % (i) + TempPath + "train.%d.pipeline.sent" % (i), shell=True)

   
    subprocess.call("python scripts//opendomain_pipeline_preSent2.py "+ DataPah + ("test.%d.coll "  % (i))  + NEResultPath + ("result.%d.NE.out > "  % (i)) + TempPath + ("test.%d.pipeline.sent" % (i)), shell=True)
   

