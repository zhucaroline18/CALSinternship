from csv import reader
import numpy as np
import requests

def load_file(filename):
    with open(filename) as file:
        datareader = reader(file)
        ret = list()
        for row in datareader:
            if row is None:
                continue
            row2 = []
            for item in row:
                row2.append(item)
            ret.append(row2)
        return ret

def fromFile(filename):
    file = load_file(filename)
    nutrients = file[0][1:]

    non_nutrients = file[2:]
    curr = 2

    ingredients = {}
    
    while(len(file[curr]) != 1):
        ingredient = file[curr[0]]
        composition = file[curr][1:]
        composition = np.float_32(composition)
        ingredients[ingredient] = composition

        curr += 1

    toReturn = {}
    while curr < len(file):
        dietName = file[curr][0]
        curr += 1

        result = np.zeros(len(nutrients))
        while(curr < len(file) and len(file[curr]) != 1):
            ingredient = file[curr][0]
            amount = eval(file[curr][1])
            result += ingredients[ingredients] * amount
            curr += 1
        toReturn[dietName] = result
    print(toReturn)
    return toReturn




# Replace with your USDA FoodData Central API key
API_KEY = 'xM53drQQ3V8CTEkvjRUwjglRJQljM1IDpvg2I1TX'

def get_nutrient_data(ingredient):
    search_url = f'https://api.nal.usda.gov/fdc/v1/foods/search?query={ingredient}&api_key={API_KEY}'
    search_response = requests.get(search_url)
    if search_response.status_code != 200:
        return f"Error: Unable to fetch data for {ingredient}"

    search_results = search_response.json()
    if 'foods' not in search_results or len(search_results['foods']) == 0:
        return f"No data found for {ingredient}"

    fdc_id = search_results['foods'][0]['fdcId']
    nutrient_url = f'https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={API_KEY}'
    nutrient_response = requests.get(nutrient_url)
    if nutrient_response.status_code != 200:
        return f"Error: Unable to fetch nutrient data for {ingredient}"

    nutrient_data = nutrient_response.json()
    nutrients = nutrient_data.get('foodNutrients', [])

    nutrient_info = {}
    for nutrient in nutrients:
        print(nutrient)
        name = nutrient['nutrient']['name']
        amount = nutrient['amount']
        unit = nutrient['nutrient']['unitName']
        nutrient_info[name] = f"{amount} {unit}"

    return nutrient_info


def nutrient_data():
    ingredient = input("Enter an ingredient: ")
    amount = input("Enter amount in grams: ")

    nutrient_info = get_nutrient_data(ingredient)
    if isinstance(nutrient_info, dict):
        print(f"Nutritional information for {ingredient}:")
        for nutrient, amount in nutrient_info.items():
            print(f"{nutrient}: {amount}")
    else:
        print(nutrient_info)

if __name__ == "__main__":
    nutrient_data()



#questions: any reccommended APIs?
#did you mean fatty acids or liipds
#There are a lot of fatty acids and lipids so are there ones in particular to look at?
#Is there a specific place to put them 