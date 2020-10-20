import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''

    indices = grades.columns.values
    full_name_to_abbv = {'lab': 'lab', 'project': 'project', 'Midterm': 'midterm', 'Final': 'final',
                         'discussion': 'disc', 'project_checkpoint': 'checkpoint'}
    output = {'lab': [], 'project': [], 'midterm': [], 'final': [], 'disc': [], 'checkpoint': []}

    for i in indices:
        cleaned = ''.join([x for x in i if not x.isdigit()])
        if cleaned in full_name_to_abbv.keys():
            output[full_name_to_abbv[cleaned]].append(i)

    return output


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''

    grades = grades.fillna(0)
    fr_str = '_free_response'
    max_str = ' - Max Points'

    totals = pd.DataFrame()
    totals['PID'] = grades['PID']

    for proj in get_assignment_names(grades)['project']:
        if proj + fr_str in grades.columns.values:
            totals[proj] = (grades[proj] + grades[proj + fr_str]) / \
                           (grades[proj + max_str] + grades[proj + fr_str + max_str])
        else:
            totals[proj] = grades[proj] / grades[proj + max_str]

    return totals.mean(axis=1)


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """

    threshold = pd.Timedelta('07:00:00')
    on_time = pd.Timedelta('00:00:00')
    late_str = ' - Lateness (H:M:S)'

    totals = pd.DataFrame()

    for lab in get_assignment_names(grades)['lab']:
        times = pd.to_timedelta(grades[lab + late_str])
        totals[lab] = (times != on_time) & (times < threshold)

    return totals.sum(axis=0)


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """

    two_week_threshold = pd.Timedelta(14, unit='D').total_seconds()
    one_week_threshold = pd.Timedelta(7, unit='D').total_seconds()
    late_threshold = pd.Timedelta('07:00:00').total_seconds()
    on_time = pd.Timedelta('00:00:00').total_seconds()

    times = pd.to_timedelta(col).dt.total_seconds()
    max_time = times.max()
    max_time = max(two_week_threshold + 1, max_time)
    return pd.cut(times, [on_time, late_threshold, one_week_threshold, two_week_threshold, max_time],
                  labels=[1.0, 0.9, 0.7, 0.4], include_lowest=True).astype('float')


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """

    late_str = ' - Lateness (H:M:S)'
    max_str = ' - Max Points'
    output = pd.DataFrame()

    for lab in get_assignment_names(grades)['lab']:
        output[lab] = grades[lab].div(grades[lab + max_str]).mul(lateness_penalty(grades[lab + late_str]))

    output.set_index(grades.index)
    return output


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """

    processed = processed.fillna(0)

    return (processed.sum(axis=1) - processed.min(axis=1)) / (len(processed.columns) - 1)


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    grades = grades.fillna(0)

    output = pd.DataFrame()

    output['project'] = projects_total(grades)
    output['lab'] = lab_total(process_labs(grades))

    assignment_names = get_assignment_names(grades)
    normal_assignments = ['disc', 'final', 'midterm', 'checkpoint']

    for key in normal_assignments:
        score_str = assignment_names[key]
        max_str = [s + ' - Max Points' for s in score_str]
        output[key] = grades[score_str].sum(axis=1) / grades[max_str].sum(axis=1)

    return (output['project'] * 0.3) + (output['lab'] * 0.2) + (output['disc'] * 0.025) + \
           (output['checkpoint'] * 0.025) + (output['midterm'] * 0.15) + (output['final'] * 0.3)


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """

    return pd.cut(total, [0, 0.6, 0.7, 0.8, 0.9, 1], labels=['F', 'D', 'C', 'B', 'A'], right=False,
                  include_lowest=True)




def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """

    return final_grades(total_points(grades)).value_counts(normalize=True)


# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of seniors
    was worse than the class under null hypothesis conditions
    (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """

    sen_grades = grades[grades['Level'] == 'SR']
    num_seniors = len(sen_grades)

    grades = total_points(grades)
    sen_grades = total_points(sen_grades)
    sen_avg = sen_grades.mean()

    avgs = []

    for i in range(N):
        sample = grades.sample(num_seniors, replace=False)
        avg = sample.mean()
        avgs.append(avg)


    return sum(np.array(avgs) < sen_avg) / N


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    assignment_names = get_assignment_names(grades)
    grades[assignment_names['lab']] = process_labs(grades)
    grades[[s + ' - Max Points' for s in assignment_names['lab']]] = 1

    assignments = []

    for category in assignment_names.values():
        for assignment in category:
            assignments.append(assignment)
    assignments_max = [s + ' - Max Points' for s in assignments]

    assignments_df = grades[assignments]
    shape = assignments_df.shape

    noise = pd.DataFrame(np.random.normal(0, 0.02, size=shape), columns=assignments_max)
    noise = noise.mul(grades[assignments_max])
    noise.columns = assignments

    grades[assignments] = grades[assignments] + noise

    grades = grades.fillna(0)

    output = pd.DataFrame()

    output['project'] = projects_total(grades)
    output['lab'] = lab_total(grades[assignment_names['lab']])

    assignment_names = get_assignment_names(grades)
    normal_assignments = ['disc', 'final', 'midterm', 'checkpoint']

    for key in normal_assignments:
        score_str = assignment_names[key]
        max_str = [s + ' - Max Points' for s in score_str]
        output[key] = grades[score_str].sum(axis=1) / grades[max_str].sum(axis=1)

    output.clip(0, 1)

    return (output['project'] * 0.3) + (output['lab'] * 0.2) + (output['disc'] * 0.025) + \
           (output['checkpoint'] * 0.025) + (output['midterm'] * 0.15) + (output['final'] * 0.3)


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """

    return [0, 77, [71, 105], 61/535, [True, False]]


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """

    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True

