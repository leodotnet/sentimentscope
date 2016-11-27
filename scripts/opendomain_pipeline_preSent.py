import optparse
import sys

# Separator of field values.
separator = '\t'


def main(sep=' '):
    #print('##Reading data from ' + result + ' and ' + test)

    file_train_test = open(sys.argv[1], 'r')

    inst = 0;

    index = 0
   
                  
    for line in file_train_test:
        line = line.strip('\n')
        line = line.strip('\r')
        if len(line.strip()) == 0 :
            print('')
            inst += 1
            index = 0
        else:
            fields = line.split(separator)
            output = line + '\t' + fields[9][0] + '\t' +  str(index)
            print(output)
            index += 1
        
            
if __name__ == '__main__':
    main(separator)
          
                    
