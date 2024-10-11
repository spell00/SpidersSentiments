import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy import stats
# Make one descriptive table of:
# - the number of pos and neg news total (1 row)
# - the number of pos and neg news per Sensationalism (2 rows)
# - Top species and genus per feeling (4 rows)
# - Top species and genus per Sensationalism (4 rows)

# Boxplots of sensationalism per Contributor, per Source, per Language, per Country,
# per Continent, per Category, per Subcategory, per Topic, per Subtopic, per Positive, per Negative
# Boxplots of Negatives per Contributor, per Source, per Language, per Country,
# per Continent, per Category, per Subcategory, per Topic, per Subtopic, per Positive, per Negative

if __name__ == "__main__":
    # Load the CSV file
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classif", type=str, default='huggingface_roberta')
    parser.add_argument("--use_preprocess", type=int, default=1)
    args = parser.parse_args()
    classif = args.classif
    use_preprocess = args.use_preprocess

    df = pd.read_csv(
        f"resultats/preprocess{use_preprocess}/resultats_sentiments_{classif}.csv", sep=','
    )
    os.makedirs(f'figures/{classif}/preprocess{use_preprocess}', exist_ok=True)
    # find correlation between sensationalism and positive and negative rates
    pos = df.iloc[:, -3]
    neg = df.iloc[:, -2]
    neu = df.iloc[:, -1]
    pval = ttest_ind(pos, neg).pvalue

    # plot frequency of positive and negative news
    plt.hist([pos, neg], bins=100, label=['Positive', 'Negative'])
    plt.title('pvalue: ' + str(pval))
    plt.legend()
    plt.savefig(f'figures/{classif}/preprocess{use_preprocess}/pos_neg_hist.png')
    plt.close()
    # Create a kde plot with curves
    sns.kdeplot([pos, neg], fill=True)
    plt.title('Frequency Plot with Curves')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.savefig(f'figures/{classif}/preprocess{use_preprocess}/pos_neg_kde.png')
    plt.close()
    print('\Total scores')

    print('\tPositive and negative defined as a news with a positive rate greater',
          'than 0.1 more than a negative news and vice-versa')

    # define a positive news as a news with a positive rate greater than 0.2
    # more than a negative news
    neutral_news = np.array([1 if n < 0.1 and p < 0.1 is True else 0 for n, p in zip(neg, pos)])
    negative_news = np.array([1 if x is True else 0 for x in neg-pos > 0.1])
    positive_news = np.array([1 if x is True else 0 for x in pos-neg > 0.1])
    print('\t\tPercentage of positive news:', positive_news.sum() / len(positive_news))
    print('\t\tPercentage of negative news:', negative_news.sum() / len(negative_news))
    print('\t\tPercentage of neutral news:', neutral_news.sum() / len(neutral_news))

    print('\tPositive and negative defined as a news with a positive rate greater than 0.2 and vice-versa')

    # define a positive news as a news with a positive rate greater than 0.2
    # more than a negative news
    negative_news = np.array([1 if x is True else 0 for x in neg > 0.2])
    positive_news = np.array([1 if x is True else 0 for x in pos > 0.2])
    neutral_news = np.array([1 if n < 0.2 and p < 0.2 is True else 0 for n, p in zip(neg, pos)])
    print('\t\tPercentage of positive news:',
          positive_news.sum() / len(positive_news)
    )
    print('\t\tPercentage of negative news:',
          negative_news.sum() / len(negative_news)
    )
    print('\t\tPercentage of neutral news:',
          neutral_news.sum() / len(neutral_news)
    )

    # Make a boxplot pos and neg scores
    plt.boxplot([pos, neg, neu])
    plt.xticks([1, 2, 3], ['Positive', 'Negative', 'Neutral'])
    plt.savefig(f'figures/{classif}/preprocess{use_preprocess}/pos_neg_neu_scores.png')
    plt.close()

    colnames = ['Sensationalism', 'Bite','Death','Figure_species',
                'Figure_bite','Expert_arachnologist','Expert_doctor',
                'Expert_others','Taxonomic_error','Venom_error',
                'Anatomy_error','Photo_error'
    ]
    pos_corr = df.loc[:, colnames].corrwith(pos, method=stats.pointbiserialr)
    neg_corr = df.loc[:, colnames].corrwith(neg, method=stats.pointbiserialr)
    neutral_corr = df.loc[:, colnames].corrwith(neu, method=stats.pointbiserialr)
    pos_corr.index = ['Correlation with positive', 'p-value with positive']
    neg_corr.index = ['Correlation with negative', 'p-value with negative']
    neutral_corr.index = ['Correlation with neutral', 'p-value with neutral']
    # concat both df
    corr = pd.concat([pos_corr, neg_corr, neutral_corr], axis=0)
    corr.to_csv(f'figures/{classif}/preprocess{use_preprocess}/correlations.csv')
    for var in colnames:
        print(f'Variable: {var}')
        var_vals = df.loc[:, var]

        print('\t\tPositive correleation:',
              var_vals.corr(pos, method=stats.pointbiserialr)
        )
        print('\t\tNegative correleation:',
              var_vals.corr(neg, method=stats.pointbiserialr)
        )
        print('\t\tNeutral correleation:',
              var_vals.corr(neu, method=stats.pointbiserialr)
        )

        print('\tPositive and negative defined as a news with a positive rate',
              'greater than 0.1 more than a negative news and vice-versa')

        # define a positive news as a news with a positive rate greater than 0.2
        # more than a negative news
        negative_news = np.array([1 if x is True else 0 for x in neg-pos > 0.1])
        positive_news = np.array([1 if x is True else 0 for x in pos-neg > 0.1])
        neutral_news = np.array([1 if n < 0.1 and p < 0.1 is True else 0 for n, p in zip(neg, pos)])
        df['Negative'] = negative_news
        df['Positive'] = positive_news
        df['Neutral'] = neutral_news

        # find correlation between sensationalism and positive and negative news
        print('\t\tPositive News correleation:',
              var_vals.corr(df.loc[:, 'Positive'], method=stats.pointbiserialr)
        )
        print('\t\tNegative News correleation:',
              var_vals.corr(df.loc[:, 'Negative'], method=stats.pointbiserialr)
        )
        print('\t\tNeutral News correleation:',
              var_vals.corr(df.loc[:, 'Neutral'], method=stats.pointbiserialr)
        )

        print('\tPositive and negative defined as a news with a positive',
              ' rate greater than 0.2 and vice-versa')

        # define a positive news as a news with a positive rate greater than 0.2
        # more than a negative news
        negative_news = np.array([1 if x is True else 0 for x in neg > 0.2])
        positive_news = np.array([1 if x is True else 0 for x in pos > 0.2])
        neutral_news = np.array([1 if n < 0.2 and p < 0.2 is True else 0 for n, p in zip(neg, pos)])

        df['Negative'] = negative_news
        df['Positive'] = positive_news
        df['Neutral'] = neutral_news

        # find correlation between sensationalism and positive and negative news
        print('\t\tPositive News correleation:',
              var_vals.corr(df.loc[:, 'Positive'], method=stats.pointbiserialr)
        )

        print('\t\tNegative News correleation:',
              var_vals.corr(df.loc[:, 'Negative'], method=stats.pointbiserialr)
        )

        print('\t\tNeutral News correleation:',
              var_vals.corr(df.loc[:, 'Neutral'], method=stats.pointbiserialr)
        )

        # Make a dataframe of only sensationalist news
        var_news = df[df[var] == 1]
        if np.isnan(var_news.loc[:, 'POS'].iloc[0]):

            pos_var = var_news.loc[:, 'POS.1']
            neg_var = var_news.loc[:, 'NEG.1']
            neu_var = var_news.loc[:, 'NEU.1']
        else:
            pos_var = var_news.loc[:, 'POS']
            neg_var = var_news.loc[:, 'NEG']
            neu_var = var_news.loc[:, 'NEU']

        # pval = ttest_ind(pos_var, neg_var).pvalue

        # Make a boxplot pos and neg scores
        plt.boxplot([pos_var, neg_var, neu_var])
        plt.xticks([1, 2, 3], ['Positive', 'Negative', 'Neutral'])
        # plt.title(f'pvalue: POSvNEG {str(pval)}')
        plt.savefig(
            f'figures/{classif}/preprocess{use_preprocess}/{var}_pos_neg_scores.png'
        )
        plt.close()

        # plot frequency of positive and negative news
        plt.hist([pos_var, neg_var, neu_var], bins=100, label=['Positive', 'Negative'])
        plt.legend()
        plt.savefig(
            f'figures/{classif}/preprocess{use_preprocess}/{var}_pos_neg_hist.png'
        )
        plt.close()
        # Create a kde plot with curves
        sns.kdeplot([pos_var, neg_var, neu_var], fill=True)
        plt.title('Frequency Plot with Curves')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.savefig(
            f'figures/{classif}/preprocess{use_preprocess}/{var}_pos_neg_kde.png'
        )
        plt.close()



# Make analysis for 'Species','Genus','Family','Order','Quality_check','Contributor'
