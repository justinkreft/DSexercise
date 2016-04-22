__author__ = 'justinkreft'

###
# Python program creates visualization png using seaborn library
#
# input=["Ex1_Flat_table.csv", "Ex1_KnnResults.csv"] from ..datafiles/
# output="Final_Output_Chart.png" to working directory
###


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    results = pd.read_csv('datafiles/Ex1_Flat_table.csv')
    results['over_50k'] = results['over_50k'].map({1: "Over 50 K", 0: "Under 50 K"})

    # Define Color Palette
    sns.set(style="white", palette="muted", color_codes=True)

    # Set up the matplotlib figure
    f, axes = plt.subplots(3, figsize=(20, 25))

    # Drop right and top spines for subplots
    sns.despine()
    sns.set_context("notebook", font_scale=1.5)

    # Create Subplot 1
    ax1 = sns.stripplot("education_num", "capital_gain", "over_50k", data=results,
                    palette="muted", size=20, marker="D",
                    edgecolor="gray", alpha=.25, ax=axes[0])
    ax1.set(ylabel="CAPITAL GAIN IN DOLLARS", xlabel="EDUCATION LEVEL", title="EDUCATION LEVEL AND CAPITAL GAIN INFLUENCE ON MAKING OVER 50K")
    ax1.legend(loc='upper left', title=" ")
    ax1.set_xticklabels(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Prof-school', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Doctorate'])

    #Create Subplot 2
    ax2 = sns.violinplot(x="over_50k", y="education_num", hue="marital_status",
                    data=results, palette="muted", inner="box",  ax=axes[1])
    ax2.set(ylabel="EDUCATION LEVEL", xlabel=" ", title="COMPARING  EDUCATION LEVEL BETWEEN MARITAL STATUS AND OVER 50K SUBGROUPS")
    ax2.legend(loc="lower center", ncol=7, title=" ")
    ax2.set_xticklabels(['Under 50K','Over 50K'])

    #Create Subplot 3
    #Pull New data
    results = pd.read_csv('Ex1_KnnResults.csv')
    results["Accuracy"] = results["over_50kActual"] + results["over_50kPredicted"]
    results['Accuracy'] = results['Accuracy'].map({1: "False Prediction", 0: "True Negative", 2: "True Positive"})

    ax3 = sns.stripplot("education_num", "capital_gain", "Accuracy", data=results,
                    palette=["#d9ef8b", "#91cf60", "#d73027"], size=20, marker="D",
                    edgecolor="gray", alpha=.25, ax=axes[2])
    ax3.set(ylabel="CAPITAL GAIN IN DOLLARS", xlabel="EDUCATION LEVEL", title="PERFORMANCE OF KNN CLASSIFIER MAPPED TO CAPITAL GAIN AND EDUCATION LEVEL ON TEST SET")
    ax3.legend(loc='upper left', title=" ")
    ax3.set_xticklabels(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Prof-school', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Doctorate'])

    #Save Figure
    plt.savefig('Final_Output_Chart.png')


main()
