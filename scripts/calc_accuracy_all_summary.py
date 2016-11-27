import sys
import string
import copy

# Separator of field values.
separator = ' '

counter_total = 0
counter_correct = 0


def main(sep=' '):
    fi = sys.stdin
    fo = sys.stdout

    result = {}

    example = 0

    for line in fi:
        line = line.strip('\n')
        #print(line)

        if line.startswith('###Stats'):

            example += 1
        elif line.startswith('###'):
            continue
        elif len(line) == 0:
            continue
        else:
            fields = line.split(":")
            score = fields[1][1:]
            if fields[0] in result:
                result[fields[0]] += float(score)
            else:
                result[fields[0]] = float(score)

    print("###Result from " + str(example) + " examples")
    for item in sorted(result):
        result[item] = result[item] / example
        print(item + ":" + str(result[item])[0:6])



if __name__ == '__main__':
    main(separator)
