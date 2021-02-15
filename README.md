# Unemployment and Mental Health

## The Problem

There have been many negative impacts of COVID-19 on our society. Two prominent effects are the increased rates of both mental health symptoms and unemployment. <a href="https://www.pewresearch.org/fact-tank/2020/05/07/a-third-of-americans-experienced-high-levels-of-psychological-distress-during-the-coronavirus-outbreak/">A Pew research study</a> found that 33% of Americans have experienced significantly high levels of psychological distress since the pandemic began. The DSM-5 states that typical prevelence for Major Depressive Disorder in the US is 7% and for Generalized Anxiety Disorder it is 2.9%. While the the Pew survey captured a broad array of symptoms and its partcipants were not diagnosed and thus the DSM statistics can't be a direct comparison, it is a helpful reference for how common anxiety and depression were in the American population before the pandemic.

According to a <a href="https://crsreports.congress.gov/product/pdf/R/R46554">report from the Congression Research Service</a> , "The unemployment rate peaked at an unprecedented level, not seen since data collection started in 1948, in April 2020 (14.7%) before declining to a still-elevated level in November(6.7%)"

Furthermore, a <a href="https://news.gallup.com/poll/171044/depression-rates-higher-among-long-term-unemployed.aspx">Gallop poll</a> indicates that those who are unemployed for 6 months or more are over three times as likely to report depression compared to those with full-time jobs.

Exploring this relationship can be tricky as the effect is bi-directional in that mental health problems can contribute to losing one's job and unemployment can cause significant psychological distress.

Better undersanding the relationship between these two areas would be helpful to helping people navigate these dfficudult times as well as mental health professionals identify patients who are most at risk for losing thier job based on thier symptomotology. There is a "chicken or the egg" problem regarding increased mental health symptoms and unemployment so no causal relationship can be ascertained from this analysis. I chose to focus on   help therapist shift focus to tools and strategies that help patients retain emplloyment.

I wanted to see what insights I could find about this relationship and if I could build a model to predict which people experiecing mental health problem were most at risk for losing thier jobs.

## The Data

The data is from a paid research study by Michael Corely, MBA, LSSBB, CPM and available on <a href="https://www.kaggle.com/michaelacorley/unemployment-and-mental-illness-survey">Kaggle</a>.

334 people were surveyed.

The survey contained a mix of yes/no, open-ended and multiple choice questions that included:
- Respondent ID
- Collector ID
- IP Address
- Email Address
- First Name
- Last Name
- I identify as having a mental illness
- Education   
- I have my own computer separate from a smart phone   
- I have been hospitalized before for my mental illness   
- How many days were you hospitalized for your mental illness   
- I am currently employed at least part-time   
- I am legally disabled   
- I have my regular access to the internet   
- I live with my parents   
- I have a gap in my resume   
- Total length of any gaps in my resume in&nbsp;months.   
- Annual income (including any social welfare programs) in USD   
- I am unemployed   
- I read outside of work and school   
- Annual income from social welfare programs   
- I receive food stamps   
- I am on section 8 housing   
- How many times were you hospitalized for your mental illness   
- Age   
- Gender   
- Household Income   
- Region   
- Device Type
- I have one of the following&nbsp;issues in addition to my illness:
    -    Lack of concentration
    -    Anxiety
    -    Depression
    -    Obsessive thinking
    -    Mood swings
    -    Panic attacks
    -    Compulsive behavior
    -    Tiredness

This data looks at a single point in time as it was captured over a period of two days, so determining the sequence of mental health issues and job loss was not possible.

### Data Cleaning and Pre-processing
The dataset needed quite a bit of cleaning and processing.
#### Drop Coumns
- IP Address, Email Address, First Name, and Last Name, had no values, which I assume was for the purposes of confidentiality
- Responent ID was unecessary since index could be used to track repondents
- CollectorID was the same for all rows so provided no useful information
#### Rename Columns
- 8 columns were ambiguously named with 'Unnamed' or 'Custom Data' labels. Inspection of the data informed more appropriate column headin

#### Null Values
In order to deal with the null values I had to convery all the columns to their appropriate datatypes as every column loaded as the 'object' dtype.

Number of nulls per column with nulls
--------------------------------------------------------------------
Lack of concentration : 283
Anxiety : 234
Depression : 248
Obsessive_thinking : 292
Mood_swings : 296
Panic_attacks : 285
Compulsive_behavior : 305
Tiredness : 234
Region : 2
How many days were you hospitalized for your mental illness : 37

The first 8 columns listed are mental health symptoms and a null value acutally represented a denial of having that symptom so these nulls were filled with 0s.

Region only contained 2 nulls so these rows were dropped.




#### Data Transfmation

#### Feature Engineering

####

## EDA

## MODELING
