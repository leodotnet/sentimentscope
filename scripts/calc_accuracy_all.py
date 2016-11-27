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



def get_predicted(predicted, answers=defaultdict(lambda: defaultdict(defaultdict))):
    global wanted_NEs
    example = 0
    word_index = 0
    entity = []
    last_ne = "O"
    last_entity = []

    answers[example] = []
    for line in predicted:
        line = line.strip()
        if line.startswith("//"):
            continue
        elif len(line) == 0:
            if entity:
                answers[example].append(list(entity))
                entity = []

            example += 1
            answers[example] = []
            word_index = 0
            continue
        else:
            split_line = line.split("\t")
            #word = split_line[0]
            value = split_line[0]
            ne = value[0]
            sent = value[2:]
            aspect = ''
            if useAspect and len(split_line) > 1:
                aspect = split_line[1]

            last_entity = []

            #check if it is start of entity
            if ne == 'B' or (ne == 'I' and last_ne == 'O'):
                if entity:
                    last_entity = list(entity)

                entity = [sent, aspect]
                    
                entity.append(word_index)

            elif ne == 'I':
                entity.append(word_index)

            elif ne == 'O':
                if last_ne == 'B' or last_ne == 'I':
                    last_entity =list(entity)
                entity = []


            if last_entity:
                answers[example].append(list(last_entity))
                last_entity = []


        last_ne = ne
        word_index += 1


    # Uncomment to norm the answers.
    #answers = norm_answers(answers)
    return answers

def get_observed(observed):
    global wanted_NEs

    example = 0
    word_index = 0
    entity = []
    last_ne = "O"
    last_entity = []

    observations=defaultdict(defaultdict)
    observations[example] = []



    for line in observed:
        line = line.strip()
        if line.startswith("//"):
            go = True
            continue
        elif len(line) == 0:
            if entity:
                observations[example].append(list(entity))
                entity = []

            example += 1
            observations[example] = []
            word_index = 0
            continue

        else:
            split_line = line.split("\t")
            word = split_line[0]
            value = split_line[9]
            ne = value[0]
            sent = value[2:]
            aspect = split_line[1][2:]

            last_entity = []

            #check if it is start of entity
            if ne == 'B' or (ne == 'I' and last_ne == 'O'):
                if entity:
                    last_entity = entity

                entity = [sent, aspect]
                    
                entity.append(word_index)

            elif ne == 'I':
                entity.append(word_index)

            elif ne == 'O':
                if last_ne == 'B' or last_ne == 'I':
                    last_entity = entity
                entity = []


            if last_entity:
                observations[example].append(list(last_entity))
                last_entity = []


        last_ne = ne
        word_index += 1



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

    subjective_hash = {"matched":0}
    subjective1_hash = {"matched":0}
    subjective0_hash = {"matched":0}

    sentiment_hash = {"true positive":0.0, "true negative":0.0, "false positive":0.0, "false negative":0.0, "matched":0}
    aspect_hash = {"true positive":0.0, "true negative":0.0, "false positive":0.0, "false negative":0.0, "matched":0}
    NE_hash = {"true positive":0.0, "true negative":0.0, "false positive":0.0, "false negative":0.0, "matched":0}


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
##        print(observed_instance)
##        print(predicted_instance)
##        print()
        total_observed += len(observed_instance)
        total_predicted += len(predicted_instance)

        for span in observed_instance:
            length = len(span) - 2
            key = 'NE_length_' + str(length)
            t = observed_hash.setdefault(key, 0)
            observed_hash[key] = t + 1

            key = 'Sent_length_' + str(length)
            t = observed_hash.setdefault(key, 0)
            observed_hash[key] = t + 1

            key = 'Subj_length_' + str(length)
            t = observed_hash.setdefault(key, 0)
            observed_hash[key] = t + 1

            key = 'Aspect_length_' + str(length)
            t = observed_hash.setdefault(key, 0)
            observed_hash[key] = t + 1

            if span[0] in wanted_sentiments:
                total_subj1_observed += 1
                

            

        for span in predicted_instance:
            length = len(span) - 2
            key = 'NE_length_' + str(length)
            t = predicted_hash.setdefault(key, 0)
            predicted_hash[key] = t + 1

            key = 'Sent_length_' + str(length)
            t = predicted_hash.setdefault(key, 0)
            predicted_hash[key] = t + 1

            key = 'Subj_length_' + str(length)
            t = predicted_hash.setdefault(key, 0)
            predicted_hash[key] = t + 1

            key = 'Aspect_length_' + str(length)
            t = predicted_hash.setdefault(key, 0)
            predicted_hash[key] = t + 1

            if span[0] in wanted_sentiments:
                total_subj1_predicted += 1



        for span in predicted_instance:
            matched = False
            span_begin = span[2]            
            span_length = len(span) - 1
            if useAspect:
                span_length = len(span) - 2
            span_ne = (span_begin, span_length)
            span_sent = span[0]
            span_subj = span_sent in wanted_sentiments
            span_aspect = span[1]
            

            for observed_span in observed_instance:
                begin = observed_span[2]
                length = len(observed_span) - 1
                if useAspect:
                    length = len(observed_span) - 2
                ne = (begin, length)
                sent = observed_span[0]
                subj = sent in wanted_sentiments
                aspect = observed_span[1]

                #NE matched
                if span_ne == ne:
                    NE_hash["matched"] += 1

                    key = 'NE_length_' + str(length)
                    t = matched_hash.setdefault(key, 0)
                    matched_hash[key] = t + 1

                    if useAspect and span_aspect == aspect:
                        aspect_hash["matched"] += 1

                        key = 'Aspect_length_' + str(length)
                        t = matched_hash.setdefault(key, 0)
                        matched_hash[key] = t + 1
                        

                    if span_sent == sent:
                        sentiment_hash["matched"] += 1

                        key = 'Sent_length_' + str(length)
                        t = matched_hash.setdefault(key, 0)
                        matched_hash[key] = t + 1

                    if span_subj == subj:
                        
                        subjective_hash["matched"] += 1

                        key = 'Subj_length_' + str(length)
                        t = matched_hash.setdefault(key, 0)
                        matched_hash[key] = t + 1
                            
                        if subj == False:                       
                   
                            subjective0_hash["matched"] += 1

                            key = 'Subj0_length_' + str(length)
                            t = matched_hash.setdefault(key, 0)
                            matched_hash[key] = t + 1
                        else:
                            subjective1_hash["matched"] += 1

                            key = 'Subj1_length_' + str(length)
                            t = matched_hash.setdefault(key, 0)
                            matched_hash[key] = t + 1
                        


    

    print('###Stats')
    print('#observed: %d' % (total_observed))
    print('#predicted: %d' % (total_predicted))

    prec = NE_hash["matched"]/total_predicted
    rec = NE_hash["matched"]/total_observed
    f = 2 * prec * rec / (prec + rec)
    print('NE precision: %.4f' % (prec))
    print('NE recall: %.4f' %   (rec))
    print('NE F: %.4f' % (f))

    prec = sentiment_hash["matched"]/total_predicted
    rec = sentiment_hash["matched"]/total_observed
    f = 2 * prec * rec / (prec + rec)

    print('Sentiment precision: %.4f' % (prec))
    print('Sentiment recall: %.4f' % (rec))
    print('Sentiment F: %.4f' % (f))

    prec = subjective_hash["matched"]/total_predicted
    rec = subjective_hash["matched"]/total_observed
    f = 2 * prec * rec / (prec + rec)
    print('Subjectivity precision: %.4f' % (prec))
    print('Subjectivity recall: %.4f' % (rec))
    print('Subjectivity F: %.4f' % (f))

    if useAspect:
        prec = aspect_hash["matched"]/total_predicted
        rec = aspect_hash["matched"]/total_observed
        f = 2 * prec * rec / (prec + rec)
        print('Aspect precision: %.4f' % (prec))
        print('Aspect recall: %.4f' % (rec))
        print('Aspect F: %.4f' % (f))




    prec = subjective1_hash["matched"]/total_subj1_predicted
    rec = subjective1_hash["matched"]/total_subj1_observed
    f = 2 * prec * rec / (prec + rec)
    print('Subjectivity1 precision: %.4f' % (prec))
    print('Subjectivity1 recall: %.4f' % (rec))
    print('Subjectivity1 F: %.4f' % (f))


    # for key in stat_hash:
    #      print(key,': ' ,stat_hash[key])

    hashitem = ['NE_length_', 'Sent_length_', 'Subj_length_',  'Subj1_length_' ]
    if useAspect:
        hashitem.append('Aspect_length_')

    if showdetail == 'Y':
        for i in range(1, 12):
            for item in hashitem:
                key = item + str(i)
                num_total_predicted = predicted_hash.setdefault(key, 0)
                num_total_observed  = observed_hash.setdefault(key, 0)
                num_matched = matched_hash.setdefault(item + str(i), 0.0)

                if num_total_predicted > 0:
                    prec = num_matched / num_total_predicted
                else:
                    prec = 0.0

                if num_total_observed > 0:
                    rec = num_matched / num_total_observed
                else:
                    rec = 0.0

                if (prec + rec > 0):
                    f = 2 * prec * rec / (prec + rec)
                else:
                    f = 0.0

                if prec > 0 or rec > 0:
                    print(key + ' precision: ' + str(prec))
                    print(key + ' recall: '+ str(rec))
                    print(key + ' F: '+ str(f))

    print('')
    #
    # print('#Item\tPrec\tRec')
    # print('NE\t%.2f\t%.2f' % (NE_hash["matched"]/total_predicted,  NE_hash["matched"]/total_observed))
    # print('Sent\t%.2f\t%.2f' % ( sentiment_hash["matched"]/total_predicted, sentiment_hash["matched"]/total_observed))
    # print('Subj\t%.2f\t%.2f' % ( subjective_hash["matched"]/total_predicted, subjective_hash["matched"]/total_observed))
    #
    #
    #

predicted = get_predicted(predicted_NE_sent)
observed = get_observed(test)
compare_observed_to_predicted(observed, predicted)
# for item in observed:
#     print(observed[item])
#
# print()
# for item in predicted:
#     print(predicted[item])
