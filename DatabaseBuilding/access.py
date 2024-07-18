import sqlite3
from csv import reader
import ast

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
insert_individual += "calcium, total_phosphorus, inorganic_available_P, caPratio, Na, Cl, K, Mg, S, Cu, I, Fe, Mn, Se, Zn)"
insert_individual += "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
insert_individual += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

insert_follower = "INSERT INTO DietFollower (diet_follower_id, pen, diet_id, consumption, total_feed_intake) VALUES (NULL, ?, (SELECT diet_id FROM Diets WHERE timePeriod = ? and species = ? and study = ? and diet_name = ?), ?, ?)"

#insert_individual_result = "INSERT INTO Results (result_id, diet_follower_id, individual_id, average_feed_intake, bodyweightgain) VALUES (NULL, NULL, (SELECT diet_id FROM Diets WHERE timePeriod = ? and species = ? and study = ? and diet_name = ?), ?, ?, ?)"

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
def load_file_sql_DietFollower(filename, begin):
    with open(filename) as file:
        datareader = reader(file)
        ret = list()
        for i, row in enumerate(datareader):
            if i == 0: continue
            if row is None:
                continue
            row2 = "("
            for i, item in enumerate(row):
                if i < begin:
                    row2 += "'" + (item) + "', "
                else:
                    row2 += item + ","
            row2 = row2[:len(row2)-1]
            row2 += ")"
            ret.append(ast.literal_eval(row2))
        return ret

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

def insertDiets(filename):
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()
    result = load_file_sql_format(filename, 5)
    print(result)

    cur.executemany(insert_diets, result)
    conn.commit()

    cur.execute('SELECT * FROM Diets')
    rows = cur.fetchall()
    for row in rows:
        print(row)
    conn.close()

def insertIndividual(filename):
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()
    result = load_file_sql_format(filename, 5)

    cur.executemany(insert_individual, result)
    conn.commit()
    conn.close()
    
def insertDietFollowers(filename):
    conn = sqlite3.connect('nutrition.db')
    cur = conn.cursor()
    toSearch = load_file_sql_DietFollower(filename, 5)

    cur.executemany(insert_follower, toSearch)
    conn.commit()
    conn.close()
    
    
if __name__ == "__main__":
    #insertDiets('diet_only.csv')
    #insertIndividual('individuals_only.csv')
    insertDietFollowers('DietFollowersOnly.csv')
