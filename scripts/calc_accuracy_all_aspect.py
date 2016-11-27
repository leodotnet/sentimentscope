import sys
import re
from copy import copy
from collections import defaultdict
from optparse import OptionParser

parser = OptionParser(usage="%prog test predictions [OPTIONS]")
parser.add_option("-l", "--linear", dest="linear", action="store_true", default=False,
                  help="Predictions are in linear format")
parser.add_option("-c", "--with-confidence",
                  action="store_true", dest="with_confidence", default=False,
                  help="Print out posteriors")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False)
parser.add_option("-t", "--top-5", action="store", dest="top_5", default=False)
(options, args) = parser.parse_args()


global wanted_NEs

wanted_NEs = ("B", "I", "B_VOLITIONAL", "I_VOLITIONAL", "B_ORGANIZATION", "I_ORGANIZATION", "B_PERSON", "I_PERSON", "Bsentiment", "Bpositive", "Bnegative", "Bneutral", "I", "Inegative", "Ipositive", "Isentiment", "Ineutral")


if len(sys.argv) < 2:
    print ("Usage:  python calc_accuracy.py test predictions")
    sys.exit()
test = open(sys.argv[1], "r")



top_5_list = []
if options.top_5:
    top_5_fid = open(options.top_5, "r").readlines()
    for line in top_5_fid:
        top_5_list += [line.strip()]

by_example = False
predicted_NE_sent = open(sys.argv[2], "r")

if options.with_confidence:
    conf_file = open("acc_conf.csv", "w+")

showdetail = 'Y'
if len(sys.argv) >= 3:
    showdetail = sys.argv[3]


useAspect = False
if len(sys.argv) >= 5:
    if sys.argv[4] == 'useAspect':
        useAspect = True

def aspect_category_equal(predicted_opinion, observed_opinion):
    return predicted_opinion[1] == observed_opinion[1]


def get_predicted(predicted, answers=defaultdict(lambda: defaultdict(defaultdict))):
    global wanted_NEs
    example = 0


    answers[example] = []
    for line in predicted:
        line = line.strip()
        if line.startswith("//"):
            continue
        else:
            split_line = line.split("|")
            answers[example] = []
            for opinion_str in split_line:
                if len(opinion_str) > 0:
                    item = opinion_str.split(',')
                    opinion = [item[0], item[1].upper(), item[2], item[3]]
                    aspect = item[1].upper()

                    Found = False
                    for other_opinion in answers[example]:
                        if aspect == other_opinion[1]:
                            Found = True
                            break
                    if not Found: 
                        answers[example].append(opinion)
            example += 1


    # Uncomment to norm the answers.
    #answers = norm_answers(answers)
    print('predicted:',example)
    return answers

def get_observed(observed):
    global wanted_NEs

    example = 0

    observations=defaultdict(defaultdict)
    observations[example] = []



    for line in observed:
        line = line.strip()
        if line.startswith("//"):
            go = True
            continue
        else:
            split_line = line.split("|")
            observations[example] = []
            for opinion_str in split_line:
                if len(opinion_str) > 0:
                    item = opinion_str.split(',')
                    opinion = [item[0], item[1], item[2], item[3]]
                    aspect = item[1]

                    Found = False
                    for other_opinion in observations[example]:
                        if aspect == other_opinion[1]:
                            Found = True
                            break
                    if not Found:
                        observations[example].append(opinion)
            example += 1
            
    print('observed:',example)
    return observations


def split_NE_sentiment(input_NE):
    NE_tag = input_NE[0]
    sent_tag = input_NE[1:]
    if NE_tag == "I":
        sent_tag = ""
    return (NE_tag, sent_tag)
    

def compare_observed_to_predicted(observed, predicted):
    #print observed
    #print predicted
    #sys.exit()
    #NE_matrix = {"B":{"B":0, "I":0, "O":0}, "I":{"B":0, "I":0, "O":0}, "O":{"B":0, "I":0, "O":0}}
    #sentiment_matrix = {"":{"":0}, "positive":{"positive":0, "negative":0, "neutral":0}, "negative":{"positive":0, "negative":0, "neutral":0}, "neutral":{"positive":0, "negative":0, "neutral":0}}
    acc_hash = {"true":0.0, "false":0.0}
    joint_hash = {"true":0.0, "false":0.0}

    target_hash = {"true positive":0.0, "true negative":0.0, "false positive":0.0, "false negative":0.0, "matched":0}    
    sentiment_hash = {"true positive":0.0, "true negative":0.0, "false positive":0.0, "false negative":0.0, "matched":0}
    aspect_hash = {"true positive":0.0, "true negative":0.0, "false positive":0.0, "false negative":0.0, "matched":0}
    #opinion_hash = {"true positive":0.0, "true negative":0.0, "false positive":0.0, "false negative":0.0, "matched":0}


    matched_hash = defaultdict(defaultdict)
    predicted_hash = defaultdict(defaultdict)
    observed_hash = defaultdict(defaultdict)



    wanted_sentiments = ("positive", "negative", "Bpositive", "Bnegative", "Bsentiment", "sentiment")

    total_observed = 0.0
    total_predicted = 0.0

    total_subj1_observed = 0.0
    total_subj1_predicted = 0.0


    for example in observed:
        # # if (example > 10):
        #     break

        observed_instance = observed[example]
        predicted_instance = predicted[example]

        total_observed += len(observed_instance)
        total_predicted += len(predicted_instance)

        #if example < 20:
        #    print(observed_instance,';\t',predicted_instance)


        for predicted_opinion in predicted_instance:
            for observed_opinion in observed_instance:
                #opinion matched
                equalAspect = False
                equalTarget = False
                if predicted_opinion[1] == observed_opinion[1]:
                    aspect_hash["matched"] += 1
                    equalAspect = True
                    #print(predicted_opinion[1], ' matched with ', observed_opinion[1])
                        
                if predicted_opinion[2] == observed_opinion[2] and  predicted_opinion[3] == observed_opinion[3]:
                    target_hash["matched"] += 1
                    equalTarget = True

                if equalAspect and equalTarget and predicted_opinion[0] == observed_opinion[0]:
                    sentiment_hash["matched"] += 1
                    

    

    print('###Stats')
    print('#observed: %d' % (total_observed))
    print('#predicted: %d' % (total_predicted))

    prec = aspect_hash["matched"]/total_predicted
    rec = aspect_hash["matched"]/total_observed
    f = 2 * prec * rec / (prec + rec)
    print('Aspect precision: %.4f' % (prec))
    print('Aspect recall: %.4f' %   (rec))
    print('Aspect F: %.4f' % (f))
    
    prec = sentiment_hash["matched"]/total_predicted
    rec = sentiment_hash["matched"]/total_observed
    f = 2 * prec * rec / (prec + rec)
    print('Polarity precision: %.4f' % (prec))
    print('Polarity recall: %.4f' %   (rec))
    print('Polarity F: %.4f' % (f))

    prec = target_hash["matched"]/total_predicted
    rec = target_hash["matched"]/total_observed
    f = 2 * prec * rec / (prec + rec)
    print('Target precision: %.4f' % (prec))
    print('Target recall: %.4f' %   (rec))
    print('Target F: %.4f' % (f))
    print('')


predicted = get_predicted(predicted_NE_sent)
observed = get_observed(test)
compare_observed_to_predicted(observed, predicted)

