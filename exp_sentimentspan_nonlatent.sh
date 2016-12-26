#!/bin/bash
cat email_succ.temp > email_succ.txt
date >> email_succ.txt
curr_date=$(date +%Y%m%d_%H%M)
seperator="_"
lang="$1"
set="$2"
beginIndex=1
endIndex=1

if [ "$set" == "full" ] 
then
beginIndex=2
endIndex=10
fi

l2=0.0005
NElen=6

if [ "$lang" == 'es' ]
then
l2=0.001
NElen=7
fi


task=$lang"_"$curr_date"_"$set

echo "set=$set task=$task begin=$beginIndex end=$endIndex"
mkdir experiments/sentiment/model/sentimentspan_nonlatent/Twitter_$lang/$task

logfile=$task".log"


echo "more "$logfile" | grep Iteration | wc -l" > check.sh
chmod +x check.sh

java -cp bin -Xmx16g com.nlp.targetedsentiment.util.TargetSentimentGlobal sentimentspan_nonlatent 1000 $l2 $beginIndex $endIndex $lang weightnotpush $task $NElen > $logfile 2>&1

python scripts/sentiment.py $beginIndex $endIndex sentimentspan_nonlatent $lang $task N

date >> email_succ.txt
#sendmail lihao.leolee@gmail.com < email_succ.txt
