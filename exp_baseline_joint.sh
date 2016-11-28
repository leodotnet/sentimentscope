#!/bin/bash
cat email_succ.temp > email_succ.txt
date >> email_succ.txt
curr_date=$(date +%Y%m%d_%H%M)
seperator="_"
lang="$1"
set="$2"
beginIndex=0
endIndex=0

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

if [ "$set" == "dev" ]
then
task="default"
fi

model="baseline_joint"

entry="-cp bin com.nlp.targetedsentiment.util.TargetSentimentGlobal"
#entry="-jar sentimentspan.jar"

echo "set=$set task=$task begin=$beginIndex end=$endIndex"
mkdir experiments/sentiment/model/$model/Twitter_$lang/$task

logfile=$task".log"

echo "java -cp bin -Xmx16g com.nlp.targetedsentiment.f.latent.TargetSentimentLearner 2000 0.0005 "$beginIndex" "$endIndex" "$lang" weightnotpush "$task" > "$logfile

echo "more "$logfile" | grep Iteration | wc -l" > check.sh
chmod +x check.sh

java -Xmx16g $entry $model 1000 $l2 $beginIndex $endIndex $lang weightnotpush $task $NElen WORD_FEATURE_OFF DUMP_FEATURE > $logfile 2>&1

#python scripts/sentiment.py $beginIndex $endIndex $model $lang $task N
date >> email_succ.txt
#sendmail lihao.leolee@gmail.com < email_succ.txt
