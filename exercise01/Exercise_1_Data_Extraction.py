__author__ = 'justinkreft'

###
# Python program that extracts data from sqlite database and flattens it into data to be analyzed.
#
# input="exercise01.sqlite" from working directory
# output="Ex1_Flat_table.csv" to ..datafiles/
###

import dataset

def main():
    #Open Database Connection
    db = dataset.connect('sqlite:///exercise01.sqlite')

    #Examine tables and columns, print for reference
    # Note: Commented out, for initial run only
    # tables = db.tables
    # print(db.tables)
    # for table in tables:
    #    print(table, db[table].columns)

    # Select Query to flatten relationships in database
    # Note: removes duplicate IDs from lookup tables and renames attributes where necessary
    query = '''SELECT records.id, age, education_num, capital_gain, capital_loss, hours_week, over_50k,
            workclasses.name AS workclass,
            education_levels.name AS education_level,
            marital_statuses.name AS marital_status,
            occupations.name AS occupation,
            relationships.name AS relationship,
            races.name AS race,
            sexes.name AS sex,
            countries.name AS country
            FROM records
            LEFT JOIN workclasses ON workclasses.id = records.workclass_id
            LEFT JOIN education_levels ON education_levels.id = records.education_level_id
            LEFT JOIN marital_statuses ON marital_statuses.id = records.marital_status_id
            LEFT JOIN occupations ON occupations.id = records.occupation_id
            LEFT JOIN relationships ON relationships.id = records.relationship_id
            LEFT JOIN races ON races.id = records.race_id
            LEFT JOIN sexes ON sexes.id = records.sex_id
            LEFT JOIN countries ON countries.id = records.country_id
            '''

    # Execute query using dataset library
    result = db.query(query)

    #Note: Make exception handling, check for nulls and check for equal count

    # Export query results to csv using dataset library
    dataset.freeze(result, format='csv', filename='datafiles/Ex1_Flat_table.csv')

main()