import subprocess
import sys

inbound = 2
outbound = 10
showdetail = 'N'
useAspect = ''

if len(sys.argv) <= 6:
    print('python sentimetn.py 1 11 model lang subpath showdetail')
    exit()

if sys.argv[1]:
    inbound = int(sys.argv[1])

if sys.argv[2]:
    outbound = int(sys.argv[2])

if sys.argv[3]:
    model = sys.argv[3]

if sys.argv[4]:
    lang = sys.argv[4]

if sys.argv[5]:
    subpath = sys.argv[5]

if sys.argv[6]:
    showdetail = sys.argv[6]

if len(sys.argv) == 8:
    if sys.argv[7] == 'useAspect':
        useAspect = 'useAspect'
    

print("Start stats: [", inbound, ",",outbound, "]")

GoldPath = 'data//Twitter_' + lang + '//'
OutputPath = 'experiments//sentiment//' + model + '//Twitter_' + lang + '//' + subpath + '//'

print('GoldPath=',GoldPath)
print('OutputPath',OutputPath)

#subprocess.call("echo ###Summary > " + OutputPath +"result.summary", shell=True)
subprocess.call("del " + OutputPath + "result.prediction.summary.all", shell=True)
subprocess.call("echo ###Summary > " + OutputPath + "result.prediction.summary.all", shell=True)

for i in range(inbound,outbound+1):
    print("python scripts//calc_accuracy_all_aspect.py " + GoldPath + "test.%d.opinions " % (i) + OutputPath + "result.%d.out " % (i)  + showdetail + ' ' + useAspect + "  >> " + OutputPath + "result.prediction.summary.all ")
    #subprocess.call("python scripts//calc_accuracy_all_aspect.py " + GoldPath + "test.%d.opinions " % (i) + OutputPath + "result.%d.out " % (i)  + showdetail + ' '  + useAspect + "  >> " + OutputPath + "result.prediction.summary.all ", shell=True)
    subprocess.call("python scripts//calc_accuracy_all_aspect.py " + GoldPath + "test.%d.opinions " % (i) + OutputPath + "result.%d.out " % (i)  + showdetail + ' '  + useAspect, shell=True)

#subprocess.call("python scripts//calc_accuracy_all_summary.py < " + OutputPath + "result.prediction.summary.all > " + OutputPath+ "result.prediction.summary", shell=True)

#OutputPath = OutputPath.replace('//', '\\')

#subprocess.call("more " + OutputPath + "result.prediction.summary", shell=True)


