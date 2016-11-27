import optparse
import sys

# Separator of field values.
separator = '\t'



def main(sep=' '):
    
    train_test_filename = sys.argv[1]

    NER_filename = sys.argv[2]
    #print('##Reading data from ' + train_test_filename + ' and ' + NER_filename)

    file_train_test = open(train_test_filename, 'r')
    file_NER = open(NER_filename, 'r')

    inst = 0
    NER = []
    NERindex = 0
    
    for line in file_NER:
        NER.append(line.strip('\n'))

    index = 0
                  
    for line in file_train_test:
        line = line.strip('\n')
        line = line.strip('\r')
        if len(line.strip()) == 0 :
            print('')
            NERindex += 1
            index = 0
        else:
            
            fields = line.split('\t')

            output = line + '\t' + NER[NERindex] + '\t' + str(index)

            # if fields[7][0] == '_':#NER[NERindex].startswith('O'):
            #     output = line + "\t" + 'O' + '\t' + str(index)
            # else:
            #     output = line + '\t' + NER[NERindex] + '\t' + str(index)
            print(output)
            NERindex += 1
            index += 1



        
            
if __name__ == '__main__':
    main(separator)

