
dealWithMissing = 0

def ANOVA(index_of_parameter, dataset):
    mapping = {}
    for row in dataset:
        tuple = (row[1], row[2], row[3], row[4])
        output = row[93:]
        if tuple in mapping:
            mapping[tuple].append(output)
        else:
            mapping[tuple] = [output]
    
    for group in mapping:
        for pen in group: 
            x
            

def load_file(filename, max_rows):
    with open(filename) as file:
        datareader = reader(file)
        ret = list()
        count = 0
        for row in datareader:
            if row is None:
                continue
            row2 = []
            for item in row:
                if item != '':
                    row2.append(float(item))
                else:
                    row2.append(dealWithMissing)
            ret.append(row2)
            count+=1
            if (max_rows > 0 and count >= max_rows):
                break
        return ret[1:]