--not sure what units beta carotine, ast, and ash and ohd3 shoudl be...
--also minerals like fe, mn, se, s, mg, k, cl etc 
insert into NutrientTypes(type, name, units)
values
    ('Carbohydrate', 'Overall_Carbohydrate', '%'),
    ('Carbohydrate', 'NDF', '%'),
    ('Carbohydrate', 'ADF', '%'),
    ('Carbohydrate', 'NFC', '%'),
    ('Carbohydrate','crude_fiber', '%'), 
    ('Carbohydrate','starch', '%'), 
    ('Protein', 'crude_protein', '%'), 
    ('Protein', 'arginine', '%'), 
    ('Protein', 'histidine', '%'), 
    ('Protein', 'isoleucine', '%'),
    ('Protein', 'leucine', '%'),
    ('Protein', 'lysine', '%'), 
    ('Protein', 'methionine', '%'), 
    ('Protein', 'phenylalanine', '%'), 
    ('Protein', 'threonine', '%'), 
    ('Protein', 'tryptophan', '%'),
    ('Protein', 'valine', '%'), 
    ('Protein', 'alanine', '%'), 
    ('Protein', 'aspartic_acid', '%'), 
    ('Protein', 'cystine', '%'), 
    ('Protein', 'met_cys', '%'), 
    ('Protein', 'glutamic_acid', '%'), 
    ('Protein', 'glycine', '%'), 
    ('Protein', 'proline', '%'), 
    ('Protein', 'serine', '%'), 
    ('Protein', 'tyrosine', '%'), 
    ('Protein', 'phe_tyr', '%'), 
    ('Fat', 'ether_extract', '%'), 
    ('Fat', 'sfa', '%'), 
    ('Fat', 'mufa', '%'), 
    ('Fat', 'pufa', '%'), 
    ('Fat', 'n3pufa', '%'), 
    ('Fat', 'n6pufa', '%'), 
    ('Fat', 'n3n6ratio', '%'), 
    ('Fat', 'c14', '%'),
    ('Fat', 'c150', '%'),
    ('Fat', 'c151', '%'),
    ('Fat', 'c160', '%'),
    ('Fat', 'c161', '%'), 
    ('Fat', 'c170', '%'), 
    ('Fat', 'c171', '%'), 
    ('Fat', 'c180', '%'), 
    ('Fat', 'c181', '%'), 
    ('Fat', 'c182cisn6la', '%'),
    ('Fat', 'c183cisn3ala', '%'), 
    ('Fat', 'c200', '%'),
    ('Fat', 'c201', '%'),
    ('Fat', 'c204n6ara', '%'), 
    ('Fat', 'c205n3epa', '%'), 
    ('Fat', 'c220', '%'), 
    ('Fat', 'c221', '%'), 
    ('Fat', 'c226n3dha', '%'), 
    ('Fat', 'c240', '%'), 
    ('Vitamin', 'ash', '%'), 
    ('Vitamin', 'vitamina', '%'), 
    ('Vitamin', 'beta_carotene', '%'), 
    ('Vitamin', 'vitamind3', 'IU/kg'), 
    ('Vitamin', 'ohd3_25', '%'), 
    ('Vitamin', 'vitamine', 'IU/kg'), 
    ('Vitamin', 'vitamink', 'ppm'), 
    ('Vitamin', 'astaxanthinast', '%'), 
    ('Vitamin', 'biotin', 'ppm'), 
    ('Vitamin', 'choline', 'ppm'), 
    ('Vitamin', 'folic_acid', 'ppm'), 
    ('Vitamin', 'niacin', 'ppm'), 
    ('Vitamin', 'pantothenic_acid', 'ppm'), 
    ('Vitamin', 'riboflavin', 'ppm'), 
    ('Vitamin', 'thiamin', 'ppm'),
    ('Vitamin', 'pyridoxine', 'ppm'), 
    ('Vitamin', 'vitaminb12', 'ppm'), 
    ('Minerals', 'calcium', '%'), 
    ('Minerals', 'total_phosphorus', '%'), 
    ('Minerals', 'inorganic_available_P', '%'), 
    ('Minerals', 'caPratio', 'Ca:P'),
    ('Minerals', 'Na', '%'), 
    ('Minerals', 'Cl', '%'), 
    ('Minerals', 'K', '%'), 
    ('Minerals', 'Mg', '%'), 
    ('Minerals', 'S', '%'), 
    ('Minerals', 'Cu', 'ppm'), 
    ('Minerals', 'I', 'ppm'), 
    ('Minerals', 'Fe', 'ppm'), 
    ('Minerals', 'Mn', 'ppm'), 
    ('Minerals', 'Se', 'ppm'), 
    ('Minerals', 'Zn', 'ppm');

insert into OutputTypes(type, name, units)
values
    ('Growth', 'average_feed_intake', '%'),
    ('Growth', 'bodyweightgain', '%'),
    ('Blood', 'akp', '%'), 
    ('Blood', 'alt', '%'), 
    ('Blood', 'gluclose', '%'), 
    ('Blood', 'nefa', '%'), 
    ('Blood', 'pip', '%'), 
    ('Blood', 'tc', '%'), 
    ('Blood', 'tg', '%'), 
    ('Blood', 'uric_acid', '%'), 
    ('Gene Breast', 'bmTOR', '%'), 
    ('Gene Breast', 'bs6k1', '%'),
    ('Gene Breast','b4ebp1', '%'),
    ('Gene Breast','bmurf1', '%'),
    ('Gene Breast','bmafbx', '%'),
    ('Gene Breast','bampk', '%'), 
    ('Gene Liver', 'lmtor', '%'),
    ('Gene Liver', 'ls6lk1', '%'),
    ('Gene Liver', 'l4ebp1', '%'),
    ('Gene Liver', 'lmurf1', '%'),
    ('Gene Liver', 'lmafbx', '%'),
    ('Gene Liver', 'lampk', '%'),
    ('Breast Meat Quality', 'bph', '%'), 
    ('Breast Meat Quality', 'bwhc', '%'), 
    ('Breast Meat Quality', 'bhardness', '%'), 
    ('Breast Meat Quality', 'bspringiness', '%'),
    ('Breast Meat Quality', 'bchewiness', '%'),
    ('Breast Meat Quality', 'bcohesiveness', '%'),
    ('Breast Meat Quality', 'bgumminess', '%'),
    ('Breast Meat Quality', 'bresilience', '%'), 
    ('Thigh Meat Quality', 'tph', '%'), 
    ('Thigh Meat Quality', 'twhc', '%'), 
    ('Thigh Meat Quality', 'thardness', '%'),
    ('Thigh Meat Quality', 'tspringiness', '%'),
    ('Thigh Meat Quality', 'tchewiness', '%');