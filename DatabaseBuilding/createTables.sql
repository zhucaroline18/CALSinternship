CREATE TABLE Diets (
    diet_id INTEGER PRIMARY KEY,
    timePeriod VARCHAR(20),
    species VARCHAR(100),
    study VARCHAR (100), 
    diet_name VARCHAR(100),
    ME_kcal_per_g FLOAT,
    Overall_Carbohydrate float, 
    NDF float, 
    ADF float, 
    NFC float,
    crude_fiber float, 
    starch float,
    crude_protein float, 
    arginine float, 
    histidine float, 
    isoleucine float,
    leucine float,
    lysine float, 
    methionine float, 
    phenylalanine float, 
    threonine float, 
    tryptophan float,
    valine float, 
    alanine float, 
    aspartic_acid float, 
    cystine float, 
    met_cys float, 
    glutamic_acid float, 
    glycine float, 
    proline float, 
    serine float, 
    tyrosine float, 
    phe_tyr float, 
    ether_extract float, 
    sfa float, 
    mufa float, 
    pufa float, 
    n3pufa float, 
    n6pufa float, 
    n3n6ratio float, 
    c14 float,
    c150 float,
    c151 FLOAT,
    c160 FLOAT,
    c161 float, 
    c170 float, 
    c171 float, 
    c180 float, 
    c181 float, 
    c182cisn6la FLOAT,
    c183cisn3ala float, 
    c200 FLOAT,
    c201 float,
    c204n6ara float, 
    c205n3epa float, 
    c220 float, 
    c221 float, 
    c226n3dha float, 
    c240 float, 
    ash float, 
    vitamina float, 
    beta_carotene float, 
    vitamind3 float, 
    ohd3_25 float, 
    vitamine float, 
    vitamink float, 
    astaxanthinast float, 
    biotin float, 
    choline float, 
    folic_acid float, 
    niacin float, 
    pantothenic_acid float, 
    riboflavin float, 
    thiamin float, pyridoxine float, 
    vitaminb12 float, 
    calcium float, 
    total_phosphorus float, 
    inorganic_available_P float, 
    caPratio float,
    Na float, 
    Cl float, 
    K float, 
    Mg float, 
    S float, 
    Cu float, 
    I float, 
    Fe float, 
    Mn float, 
    Se float, 
    Zn float
);

CREATE TABLE Results (
    result_id INTEGER PRIMARY KEY,
    diet_folower_id Integer references DietFollower,
    individual_id INTEGER references Individuals,
    average_feed_intake float,
    bodyweightgain float,
    akp float, 
    alt float, 
    gluclose float, 
    nefa float, 
    pip float, 
    tc float, 
    tg float, 
    uric_acid float, 
    bmTOR float, 
    bs6k1 float,
    b4ebp1 float,
    bmurf1 float,
    bmafbx float,
    bampk FLOAT, 
    lmtor float,
    ls6lk1 float,
    l4ebp1 float,
    lmurf1 float,
    lmafbx float,
    lampk float,
    bph float, 
    bwhc float, 
    bhardness float, 
    bspringiness float,
    bchewiness float,
    bcohesiveness float,
    bgumminess float,
    bresilience float, 
    tph float, 
    twhc float, 
    thardness float,
    tspringiness float,
    tchewiness float
);

CREATE TABLE OutputTypes (
    type varchar(100),
    name varchar(100), 
    units varchar(100)
);

CREATE TABLE DietFollower (
    diet_follower_id INTEGER PRIMARY KEY,
    diet_id INTEGER REFERENCES Diets,
    pen VARCHAR(20), 
    consumption FLOAT,
    total_feed_intake FLOAT
);

CREATE TABLE Individuals (
    individual_id INTEGER PRIMARY KEY,
    pen VARCHAR(20),
    timePeriod VARCHAR(20),
    species VARCHAR(100),
    study VARCHAR (100), 
    diet_name VARCHAR(100),
    consumption FLOAT, 
    total_feed_intake FLOAT,
    ME_kcal_per_g FLOAT,
    Overall_Carbohydrate float, 
    NDF float, 
    ADF float, 
    NFC float,
    crude_fiber float, 
    starch float,
    crude_protein float, 
    arginine float, 
    histidine float, 
    isoleucine float,
    leucine float,
    lysine float, 
    methionine float, 
    phenylalanine float, 
    threonine float, 
    tryptophan float,
    valine float, 
    alanine float, 
    aspartic_acid float, 
    cystine float, 
    met_cys float, 
    glutamic_acid float, 
    glycine float, 
    proline float, 
    serine float, 
    tyrosine float, 
    phe_tyr float, 
    ether_extract float, 
    sfa float, 
    mufa float, 
    pufa float, 
    n3pufa float, 
    n6pufa float, 
    n3n6ratio float, 
    c14 float,
    c150 float,
    c151 FLOAT,
    c160 FLOAT,
    c161 float, 
    c170 float, 
    c171 float, 
    c180 float, 
    c181 float, 
    c182cisn6la FLOAT,
    c183cisn3ala float, 
    c200 FLOAT,
    c201 float,
    c204n6ara float, 
    c205n3epa float, 
    c220 float, 
    c221 float, 
    c226n3dha float, 
    c240 float, 
    ash float, 
    vitamina float, 
    beta_carotene float, 
    vitamind3 float, 
    ohd3_25 float, 
    vitamine float, 
    vitamink float, 
    astaxanthinast float, 
    biotin float, 
    choline float, 
    folic_acid float, 
    niacin float, 
    pantothenic_acid float, 
    riboflavin float, 
    thiamin float, pyridoxine float, 
    vitaminb12 float, 
    calcium float, 
    total_phosphorus float, 
    inorganic_available_P float, 
    caPratio float,
    Na float, 
    Cl float, 
    K float, 
    Mg float, 
    S float, 
    Cu float, 
    I float, 
    Fe float, 
    Mn float, 
    Se float, 
    Zn float
);


CREATE TABLE NutrientTypes (
    type VARCHAR(100), 
    name VARCHAR(100), 
    units VARCHAR(100)
);
