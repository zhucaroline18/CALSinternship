import sqlite3
from csv import reader
import ast
import itertools 


insert_diets = "INSERT INTO Diets (diet_id, timePeriod ,species, study, diet_name, ME_kcal_per_g,"
insert_diets += "Overall_Carbohydrate, NDF, ADF, NFC, crude_fiber, starch," 
insert_diets += "crude_protein, arginine, histidine, isoleucine,leucine ,lysine, methionine, phenylalanine, threonine, tryptophan,valine, alanine, aspartic_acid, cystine, met_cys, glutamic_acid, glycine, proline, serine, tyrosine, phe_tyr,"
insert_diets += "ether_extract, sfa, mufa, pufa, n3pufa, n6pufa, n3n6ratio, c14,c150,c151,c160,c161, c170, c171, c180, c181, c182cisn6la,c183cisn3ala, c200, c201,c204n6ara, c205n3epa, c220, c221, c226n3dha, c240, "
insert_diets += "ash, vitamina, beta_carotene, vitamind3, ohd3_25, vitamine, vitamink, astaxanthinast, biotin, choline, folic_acid, niacin, pantothenic_acid, riboflavin, thiamin, pyridoxine, vitaminb12, " 
insert_diets += "calcium, total_phosphorus, inorganic_available_P, caPratio, Na, Cl, K, Mg, S, Cu, I, Fe, Mn, Se, Zn)"
insert_diets += "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_diets += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_diets += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_diets += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_diets += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_diets += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_diets += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_diets += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_diets += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

insert_individual = "INSERT INTO Individuals (individual_id, pen, timePeriod ,species, study, diet_name, consumption, total_feed_intake, ME_kcal_per_g,"
insert_individual += "Overall_Carbohydrate, NDF, ADF, NFC, crude_fiber, starch," 
insert_individual += "crude_protein, arginine, histidine, isoleucine,leucine ,lysine, methionine, phenylalanine, threonine, tryptophan,valine, alanine, aspartic_acid, cystine, met_cys, glutamic_acid, glycine, proline, serine, tyrosine, phe_tyr,"
insert_individual += "ether_extract, sfa, mufa, pufa, n3pufa, n6pufa, n3n6ratio, c14,c150,c151,c160,c161, c170, c171, c180, c181, c182cisn6la,c183cisn3ala, c200, c201,c204n6ara, c205n3epa, c220, c221, c226n3dha, c240, "
insert_individual += "ash, vitamina, beta_carotene, vitamind3, ohd3_25, vitamine, vitamink, astaxanthinast, biotin, choline, folic_acid, niacin, pantothenic_acid, riboflavin, thiamin, pyridoxine, vitaminb12, " 
insert_individual += "calcium, total_phosphorus, inorganic_available_P, caPratio, Na, Cl, K, Mg, S, Cu, I, Fe, Mn, Se, Zn, result_id)"
insert_individual += "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "

insert_follower = "INSERT INTO DietFollower (diet_follower_id, pen, diet_id, consumption, total_feed_intake, result_id) VALUES (NULL, ?, (SELECT diet_id FROM Diets WHERE timePeriod = ? and species = ? and study = ? and diet_name = ?), ?, ?, "

insert_result = "INSERT INTO RESULTS(result_id, average_feed_intake, bodyweightgain, akp, alt, glucose, nefa, pip, tc, tg, trap, uric_acid, bmTOR, bs6k1, b4ebp1, bmurf1, bmafbx, bampk, lmtor, ls6lk1, l4ebp1, lmurf1, lmafbx, lampk, bph, bwhc, bhardness, bspringiness, bchewiness, bcohesiveness, bgumminess, bresilience, tph, twhc, thardness, tspringiness, tchewiness)"
insert_result += "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
insert_result += '?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '
insert_result += '?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'

#getSimpleIndividual = "SELECT individual_id, pen, timePeriod, vitamine, results.akp FROM individuals JOIN results on individuals.result_id = results.result_id"

getAllDietFollower = 'SELECT (d.ME_kcal_per_g) /1000 * df.total_feed_intake, '
getAllDietFollower += 'df.total_feed_intake * d.NDF, df.total_feed_intake * d.ADF, df.total_feed_intake * d.NFC, df.total_feed_intake * d.crude_fiber, df.total_feed_intake * d.starch, '
getAllDietFollower += "df.total_feed_intake * d.crude_protein, df.total_feed_intake * d.arginine, df.total_feed_intake * d.histidine, df.total_feed_intake * d.isoleucine,df.total_feed_intake * d.leucine ,df.total_feed_intake * d.lysine, df.total_feed_intake * d.methionine, df.total_feed_intake * d.phenylalanine, df.total_feed_intake * d.threonine, df.total_feed_intake * d.tryptophan,df.total_feed_intake * d.valine, df.total_feed_intake * d.alanine, df.total_feed_intake * d.aspartic_acid, df.total_feed_intake * d.cystine, df.total_feed_intake * d.met_cys, df.total_feed_intake * d.glutamic_acid, df.total_feed_intake * d.glycine, df.total_feed_intake * d.proline, df.total_feed_intake * d.serine, df.total_feed_intake * d.tyrosine, df.total_feed_intake * d.phe_tyr, "
getAllDietFollower += "df.total_feed_intake * d.ether_extract, df.total_feed_intake * d.sfa, df.total_feed_intake * d.mufa, df.total_feed_intake * d.pufa, df.total_feed_intake * d.n3pufa, df.total_feed_intake * d.n6pufa, df.total_feed_intake * d.n3n6ratio, df.total_feed_intake * d.c14,df.total_feed_intake * d.c150,df.total_feed_intake * d.c151,df.total_feed_intake * d.c160,df.total_feed_intake * d.c161, df.total_feed_intake * d.c170, df.total_feed_intake * d.c171, df.total_feed_intake * d.c180, df.total_feed_intake * d.c181, df.total_feed_intake * d.c182cisn6la,df.total_feed_intake * d.c183cisn3ala, df.total_feed_intake * d.c200, df.total_feed_intake * d.c201,c204n6ara, df.total_feed_intake * d.c205n3epa, df.total_feed_intake * d.c220, df.total_feed_intake * d.c221, df.total_feed_intake * d.c226n3dha, df.total_feed_intake * d.c240, "
getAllDietFollower += "df.total_feed_intake * d.ash, df.total_feed_intake * d.vitamina, df.total_feed_intake * d.beta_carotene, df.total_feed_intake * d.vitamind3, df.total_feed_intake * d.ohd3_25, df.total_feed_intake * d.vitamine, df.total_feed_intake * d.vitamink, df.total_feed_intake * d.astaxanthinast, df.total_feed_intake * d.biotin, df.total_feed_intake * d.choline, df.total_feed_intake * d.folic_acid, df.total_feed_intake * d.niacin, df.total_feed_intake * d.pantothenic_acid, df.total_feed_intake * d.riboflavin, df.total_feed_intake * d.thiamin, df.total_feed_intake * d.pyridoxine, df.total_feed_intake * d.vitaminb12, " 
getAllDietFollower += "df.total_feed_intake * d.calcium, df.total_feed_intake * d.total_phosphorus, df.total_feed_intake * d.inorganic_available_P, df.total_feed_intake * d.caPratio, df.total_feed_intake * d.Na, df.total_feed_intake * d.Cl, df.total_feed_intake * d.K, df.total_feed_intake * d.Mg, df.total_feed_intake * d.S, df.total_feed_intake * d.Cu, df.total_feed_intake * d.I, df.total_feed_intake * d.Fe, df.total_feed_intake * d.Mn, df.total_feed_intake * d.Se, df.total_feed_intake * d.Zn, "
getAllDietFollower += 'results.average_feed_intake, results.bodyweightgain, results.akp, results.alt, results.glucose, results.nefa, results.pip, results.tc, results.tg, results.trap, results.uric_acid, results.bmTOR, results.bs6k1, results.b4ebp1, results.bmurf1, results.bmafbx, results.bampk, results.lmtor, results.ls6lk1, results.l4ebp1, results.lmurf1, results.lmafbx, results.lampk, results.bph, results.bwhc, results.bhardness, results.bspringiness, results.bchewiness, results.bcohesiveness, results.bgumminess, results.bresilience, results.tph, results.twhc, results.thardness, results.tspringiness, results.tchewiness '
getAllDietFollower += 'FROM DietFollower AS df JOIN results on df.result_id = results.result_id JOIN diets AS d on d.diet_id = df.diet_id '

getAllIndividual = 'SELECT ME_kcal_per_g, Overall_Carbohydrate, NDF, ADF, NFC, crude_fiber, starch,'
getAllIndividual += "crude_protein, arginine, histidine, isoleucine,leucine ,lysine, methionine, phenylalanine, threonine, tryptophan,valine, alanine, aspartic_acid, cystine, met_cys, glutamic_acid, glycine, proline, serine, tyrosine, phe_tyr,"
getAllIndividual += "ether_extract, sfa, mufa, pufa, n3pufa, n6pufa, n3n6ratio, c14,c150,c151,c160,c161, c170, c171, c180, c181, c182cisn6la,c183cisn3ala, c200, c201,c204n6ara, c205n3epa, c220, c221, c226n3dha, c240, "
getAllIndividual += "ash, vitamina, beta_carotene, vitamind3, ohd3_25, vitamine, vitamink, astaxanthinast, biotin, choline, folic_acid, niacin, pantothenic_acid, riboflavin, thiamin, pyridoxine, vitaminb12, " 
getAllIndividual += "calcium, total_phosphorus, inorganic_available_P, caPratio, Na, Cl, K, Mg, S, Cu, I, Fe, Mn, Se, Zn, "
getAllIndividual += 'results.average_feed_intake, results.bodyweightgain, results.akp, results.alt, results.glucose, results.nefa, results.pip, results.tc, results.tg, results.trap, results.uric_acid, results.bmTOR, results.bs6k1, results.b4ebp1, results.bmurf1, results.bmafbx, results.bampk, results.lmtor, results.ls6lk1, results.l4ebp1, results.lmurf1, results.lmafbx, results.lampk, results.bph, results.bwhc, results.bhardness, results.bspringiness, results.bchewiness, results.bcohesiveness, results.bgumminess, results.bresilience, results.tph, results.twhc, results.thardness, results.tspringiness, results.tchewiness '
getAllIndividual += 'FROM individuals JOIN results on individuals.result_id = results.result_id '


#loads the file in an array form
def load_file(filename):
    with open(filename) as file:
        datareader = reader(file)
        ret = list()
        for row in datareader:
            if row is None:
                continue
            row2 = []
            for i, item in enumerate(row):
                if i < 4:
                    row2.append(item)
                elif item != '':
                    row2.append(float(item))
                else:
                    row2.append(0)
            ret.append(row2)
        return ret[1:]

#loads the file into an array of tuples to be used
def load_file_sql_DietFollower(filename, begin, start):
    with open(filename) as file:
        datareader = reader(file)
        ret = list()
        ret2 = list()
        for i, row in enumerate(datareader):
            if i == 0: continue
            if row is None:
                continue
            row2 = "("
            row3 = "("
            for i, item in enumerate(row):
                if i < start:
                    if i < begin:
                        row2 += "'" + (item) + "', "
                    else:
                        row2 += item + ","
                else:
                    if item != '':
                        row3 += item + ", "
                    else:
                        row3 += "NULL, "

            row2 = row2[:len(row2)-1]
            row2 += ")"
            row3 = row3[:len(row3)-1]
            row3 += ")"
            ret.append(ast.literal_eval(row2))
            ret2.append(ast.literal_eval(row3))
        return ret, ret2

#load the diet information into sql tuples
def load_file_sql_format(filename, begin_numbers):
    with open(filename) as file:
        datareader = reader(file)
        ret = list()
        for i, row in enumerate(datareader):
            if i == 0: continue
            if row is None:
                continue
            row2 = "("
            for i, item in enumerate(row):
                if i < begin_numbers:
                    row2 += "'" + (item) + "', "
                elif item != '':
                    row2 += item + ","
                else:
                    row2 += '0, '
            row2 = row2[:len(row2)-2]
            row2 += ")"
            ret.append(ast.literal_eval(row2))
        return ret
    
#load the individual data into sql tuples with the input and output data
def load_file_sql_format_individual(filename, begin_numbers, beginOutput):
    with open(filename) as file:
        datareader = reader(file)
        ret = list()
        ret2 = list()
        for i, row in enumerate(datareader):
            if i == 0: continue
            if row is None:
                continue
            row2 = "("
            row3 = "("
            for i, item in enumerate(row):
                if i < beginOutput:
                    if i < begin_numbers:
                        row2 += "'" + (item) + "', "
                    elif item != '':
                        row2 += item + ","
                    else:
                        row2 += '0, '
                else:
                    if item != '':
                        row3 += item + ","
                    else:
                        row3 += 'NULL, '
            row2 = row2[:len(row2)-2]
            row3 = row3[:len(row3)-2]
            row2 += ")"
            row3 += ")"
            ret2.append(ast.literal_eval(row3))
            ret.append(ast.literal_eval(row2))
        return ret, ret2

def insertDiets(filename):
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()
    result = load_file_sql_format(filename, 5)

    cur.executemany(insert_diets, result)
    conn.commit()

    cur.execute('SELECT * FROM Diets')
    rows = cur.fetchall()
    for row in rows:
        print(row)
    
    conn.close()

def insertIndividual(filename, start):
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()
    toSearch, result = load_file_sql_format_individual(filename, 5, start)

    for (query, res) in zip(toSearch, result):
        cur.execute(insert_result, res)
        last_id = cur.lastrowid
        cur.execute(getToInsertIndividual(last_id), query)
        
    conn.commit()
    cur.execute('SELECT * FROM individuals')
    rows = cur.fetchall()
    for row in rows:
        print(row)
    conn.close()

def getToInsertString(lastId):
    x = insert_follower + str(lastId) + ")"
    return x

def getToInsertIndividual(lastId):
    x = insert_individual + str(lastId) + ")"
    return x

def insertDietFollowers(filename, begin):
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()
    cur.execute('DELETE FROM dietFollower')
    toSearch, result = load_file_sql_DietFollower(filename, 5, begin)

    for (query, res) in zip(toSearch, result):
        cur.execute(insert_result, res)
        last_id = cur.lastrowid
        cur.execute(getToInsertString(last_id), query)

    conn.commit()
    cur.execute('SELECT * FROM dietFollower')
    rows = cur.fetchall()
    for row in rows:
        print(row)
    conn.close()

def searchSpecificAnimal(animal):
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()
    search = "WHERE species = '" + animal + "'"
    cur.execute(getAllIndividual + search)
    toReturn = cur.fetchall()

    search = "WHERE d.species = '" + animal + "'"
    cur.execute(getAllDietFollower + search)
    toReturn.append(cur.fetchall())
    conn.close()
    return toReturn

def searchSpecificNutrient(nutrient):
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()
    search = "WHERE " + nutrient + " != 0"
    cur.execute(getAllIndividual + search)
    toReturn = cur.fetchall()

    search = "WHERE d." + nutrient + " != 0"
    cur.execute(getAllDietFollower + search)
    toReturn.append(cur.fetchall())
    conn.close()
    return toReturn

def searchStudies(studies):
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()
    toReturn = []
    for study in studies:
        search = "WHERE study = '" + study + "'"
        print(search)
        print(getAllIndividual + search)
        cur.execute(getAllIndividual + search)
        toReturn.append(cur.fetchall())

        search = "WHERE d.study = '" + study + "'"
        cur.execute(getAllDietFollower + search)
        toReturn.append(cur.fetchall())
    conn.close()
    return toReturn

if __name__ == "__main__":
    #insertDiets('diet_only.csv')
    insertIndividual('individuals2.csv', 93)
    insertDietFollowers('dietFollowers2.csv', 7)
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()

    #print(searchStudies(["broiler DHA"]))
    #print(searchSpecificNutrient('calcium'))

    '''cur.execute(getAllDietFollower)
    rows = cur.fetchall()
    for row in rows:
        print(row)
    conn.close()'''

