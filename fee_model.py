import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo 
    import pandas as pd 
    import numpy as np
    import altair as alt
    import os 
    return alt, mo, np, os, pd


@app.cell
def _(mo, os, pd):
    fee_data = pd.read_excel(os.path.join(mo.notebook_location(),'public/fee_data.xlsx'))
    return (fee_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Simulation of PAYT reform

    In this notebook we can simulate impacts of waste collection reforms on tax revenue and expenditures of OLO. 
    We will adjust different parameters to show the expected evolution of key indicators in various scenarios. Since we have little available past data on how people in Bratislava respond to changes in waste collection, we need to model different scenarios to show a potential range of outcomes. 

    Here you can set up the simulation: 

    1. Increase in fees (in %)
    2. Removing / adding options of collection schedules

    The simulation is parametrized differently for individual homes and differently for businesses/coops. This is because the assumption is that these groups respond differently to fee increases. While individual home owners can respond by frequency changes only (usually only have on bin), coops and businesses mostly adjust the number of bins, and only if this is not an option do they drop frequencies. 

    You can setup a simple model (one % fee increase) or a stepped model where we can schedule several adjustments to fees over time.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pd):
    fee_hike = mo.ui.slider(0,100, show_value=True, include_input=True, value=30)
    remove_weekly = mo.ui.checkbox()
    forecast_periods = mo.ui.slider(0,20, show_value=True, include_input=True, value=5)

    ind_setup_df = pd.DataFrame(columns=['Fee Hike','Year Gap','Disable Weekly Option'], data=[[30,5,False]])
    ind_setup_editor = mo.ui.data_editor(data=ind_setup_df)

    fee_hike_coop = mo.ui.slider(0,100, show_value=True, include_input=True, value=30)
    forecast_periods_coop = mo.ui.slider(0,20, show_value=True, include_input=True, value=5)

    coop_setup_df = pd.DataFrame(columns=['Fee Hike','Year Gap'], data=[[30,5]])
    coop_setup_editor = mo.ui.data_editor(data=coop_setup_df)

    mo.vstack([
        mo.md('## Parameter setup'),
        mo.md('### Individual homes (simple setup)'),
        mo.hstack([mo.md('**1 Increase in standard fee in %**'), fee_hike], align='center'),
        mo.hstack([mo.md('**2 Disable once per week frequency option for individual homes**'), remove_weekly],  align='center'),
        mo.hstack([mo.md('**3 Number of years to forecast**'), forecast_periods], align='center'),
        mo.md('### Individual homes (complex setup)'),
        mo.md('In this table, you can add multiple rows as steps in gradual fee increase / change'),
        ind_setup_editor,
        mo.md('### Coops/businesses (simple setup)'),
        mo.hstack([mo.md('**1 Increase in standard fee in %**'), fee_hike_coop], align='center'),
        mo.hstack([mo.md('**2 Number of years to forecast**'), forecast_periods_coop], align='center'),
        mo.md('### Coops/businesses (complex setup)'),
        coop_setup_editor

    ])


    return (
        coop_setup_editor,
        fee_hike,
        fee_hike_coop,
        forecast_periods,
        forecast_periods_coop,
        ind_setup_editor,
        remove_weekly,
    )


@app.cell(hide_code=True)
def _(alt, ind_df_results, mo):
    ind_fee_chart = mo.ui.altair_chart(alt.Chart(ind_df_results).mark_trail().encode(x='Period', y="Individual payers fees", color='Scenario'), label="Evolution of fees from individual payers in EUR")

    ind_bin_chart = mo.ui.altair_chart(alt.Chart(ind_df_results).mark_trail().encode(x='Period', y="Individual bin count", color='Scenario'), label="Evolution of # of bins from individual payers")

    """
    mo.vstack([
        ind_bin_chart,
        ind_fee_chart
    ])
    """
    ind_fee_chart
    return


@app.cell
def _(alt, coop_results_, mo):
    coop_fee_chart = mo.ui.altair_chart(alt.Chart(coop_results_).mark_trail().encode(x='Period', y="Coop fees", color='Scenario'), label="Evolution of fees from coop and business payers in EUR")

    coop_bin_chart = mo.ui.altair_chart(alt.Chart(coop_results_).mark_trail().encode(x='Period', y="Coop bin count", color='Scenario'), label="Evolution of # of bins for coop and business payers")

    mo.vstack([
        coop_fee_chart,
        coop_bin_chart
    ])
    return


@app.cell
def _(fee_data):
    # basic groupings of current data
    grouped_overview = fee_data.groupby(['Year','Payer','CapacityInt'], as_index=False)[['BinCount','TotalFee','TotalWeeklyVolume','CollectionPoints']].sum()
    bin_count_overview = grouped_overview.pivot(columns='Year',index=['Payer','CapacityInt'],values=['BinCount'])
    fee_overview = grouped_overview.pivot(columns='Year',index=['Payer','CapacityInt'],values=['TotalFee'])
    volume_overview = grouped_overview.pivot(columns='Year',index=['Payer','CapacityInt'],values=['TotalWeeklyVolume'])

    big_grouped_overview = fee_data.groupby(['Year','Payer'], as_index=False)[['BinCount','TotalFee','TotalWeeklyVolume','CollectionPoints']].sum()

    big_grouped_overview['BinsPerPoint'] = big_grouped_overview.BinCount.div(big_grouped_overview.CollectionPoints)
    grouped_overview['BinsPerPoint'] = grouped_overview.BinCount.div(grouped_overview.CollectionPoints)
    grouped_overview['PayerCapacity'] = grouped_overview.Payer + grouped_overview.CapacityInt.astype(str)
    return


@app.cell
def _(fee_data, fee_hike_coop):
    # get baseline for coops and businesses homes
    coop_sample = fee_data.query('Payer != "Individual"')
    coop_baseline = coop_sample.query('Year == 2025').groupby(['IntervalPerWeek'], as_index=False)[['CollectionPoints','BinCount','TotalFee','TotalVolume']].sum()
    coop_baseline['FeePerBin'] = coop_baseline['TotalFee'].div(coop_baseline['BinCount'])
    coop_baseline['BinPerPoint'] = coop_baseline['BinCount'].div(coop_baseline['CollectionPoints'])

    coop_old_fees = coop_baseline[['FeePerBin']].values
    coop_old_bin_ratio = coop_baseline[['BinPerPoint']].values
    coop_old_points = coop_baseline[['CollectionPoints']].values

    coop_fee_hike_pct = fee_hike_coop.value/100. 
    coop_new_fees = coop_old_fees * (1 + coop_fee_hike_pct)
    return (
        coop_fee_hike_pct,
        coop_new_fees,
        coop_old_bin_ratio,
        coop_old_fees,
        coop_old_points,
    )


@app.cell
def _(
    bin_per_point_forecast,
    coop_fee_hike_pct,
    coop_new_fees,
    coop_old_bin_ratio,
    coop_old_fees,
    coop_old_points,
    forecast_periods_coop,
    np,
    pd,
):
    simple_coop_forecast_bins, latest_sample = bin_per_point_forecast(coop_old_bin_ratio, coop_old_points, coop_fee_hike_pct, forecast_periods_coop.value)
    simple_coop_forecast_fees = simple_coop_forecast_bins * coop_new_fees.T
    simple_coop_forecast_fees[0] = simple_coop_forecast_bins[0] * coop_old_fees.T 
    simple_coop_forecast_fees

    coop_results = pd.DataFrame({'Coop fees': simple_coop_forecast_fees.sum(axis=1),
                               'Coop bin count': simple_coop_forecast_bins.sum(axis=1),
                               'Period': np.arange(forecast_periods_coop.value),
                                'Scenario': 'Simple'})
    return (coop_results,)


@app.cell
def _(coop_results, np, pd, stepped_coop):
    stepped_coop_bins, stepped_coop_fees = stepped_coop()
    stepped_coop_bins_total = stepped_coop_bins.sum(axis=1)
    stepped_coop_fees_total = stepped_coop_fees.sum(axis=1)

    coop_results2 = pd.DataFrame(
        {'Coop fees': stepped_coop_fees_total,
         'Coop bin count': stepped_coop_bins_total,
         'Period': np.arange(len(stepped_coop_fees_total)),
         'Scenario': 'Stepped model'})

    coop_results_ = pd.concat([coop_results, coop_results2], ignore_index=True)
    return (coop_results_,)


@app.cell
def _(fee_data):
    # get baseline for individual homes
    individual_sample = fee_data.query('Payer == "Individual" and IntervalPerWeek < 2 and CapacityInt < 1000')
    ind_baseline = individual_sample.query('Year == 2025')[['CapacityInt','IntervalPerWeek','CollectionPoints','BinCount','TotalFee','TotalVolume']]
    ind_baseline['FeePerBin'] = ind_baseline['TotalFee'].div(ind_baseline['BinCount'])
    return (ind_baseline,)


@app.cell
def _(fee_hike, ind_baseline):
    # get baseline for bins and fees for homes
    ind_fee_hike_pct = fee_hike.value/100.
    ind_old_fees = ind_baseline[['FeePerBin']].values 
    ind_baseline_bins = ind_baseline[['BinCount']].values.T
    ind_new_fees = ind_old_fees * (1. + ind_fee_hike_pct)
    return ind_baseline_bins, ind_fee_hike_pct, ind_new_fees, ind_old_fees


@app.cell
def _(
    forecast_periods,
    ind_baseline_bins,
    ind_fee_hike_pct,
    ind_new_fees,
    ind_old_fees,
    remove_weekly,
    run_individual,
):
    # forecast individual homes
    ind_bin_evolution = run_individual(ind_baseline_bins, forecast_periods.value, 0, price_increase=ind_fee_hike_pct, remove_weekly=remove_weekly.value)
    ind_fee_evolution = ind_bin_evolution * ind_new_fees.T
    ind_fee_evolution[0] = ind_bin_evolution[0] * ind_old_fees.T # first year keeps old fees
    return ind_bin_evolution, ind_fee_evolution


@app.cell
def _(ind_bin_evolution, ind_fee_evolution):
    ind_fee_evolution_total = ind_fee_evolution.sum(axis=1)
    ind_bin_evolution_total = ind_bin_evolution.sum(axis=1)
    return ind_bin_evolution_total, ind_fee_evolution_total


@app.cell
def _(ind_baseline_bins, ind_old_fees, ind_setup_editor, np, run_individual):
    def stepped_individual():

        old_fees = ind_old_fees.copy() 
        baseline_bins = ind_baseline_bins.copy() 
        bin_evolution = np.array([])
        fee_evolution = np.array([])
        price_hike = 0.

        for step in ind_setup_editor.value.iterrows():
            step_values = step[1]
            price_hike = price_hike + (step_values['Fee Hike'] / 100.)
            _new_fees = old_fees * (1 + price_hike)
            _evolution = run_individual(baseline_bins, step_values['Year Gap'], 0, price_increase = price_hike, remove_weekly = step_values['Disable Weekly Option'])
            _fee_evolution = _evolution * _new_fees.T 
            if len(bin_evolution) == 0:
                _fee_evolution[0] = _evolution[0] * old_fees.T
                bin_evolution = _evolution
                fee_evolution = _fee_evolution
            else:
                _fee_evolution = _fee_evolution[1:]
                _evolution = _evolution[1:]
                bin_evolution = np.concatenate([bin_evolution, _evolution])
                fee_evolution = np.concatenate([fee_evolution, _fee_evolution])
            baseline_bins = np.array([_evolution[-1]])

        return bin_evolution, fee_evolution

    stepped_ind_bins, stepped_ind_fees = stepped_individual()
    stepped_ind_bins_total = stepped_ind_bins.sum(axis=1)
    stepped_ind_fees_total = stepped_ind_fees.sum(axis=1)
    return stepped_ind_bins_total, stepped_ind_fees_total


@app.cell
def _(
    forecast_periods,
    ind_bin_evolution_total,
    ind_fee_evolution_total,
    np,
    pd,
    stepped_ind_bins_total,
    stepped_ind_fees_total,
):
    _ind_df_results = pd.DataFrame(
        {'Individual payers fees': ind_fee_evolution_total,
         'Individual bin count': ind_bin_evolution_total,
         'Period': np.arange(forecast_periods.value+1),
         'Scenario': 'Simple model'})

    _ind_df_results2 = pd.DataFrame(
        {'Individual payers fees': stepped_ind_fees_total,
         'Individual bin count': stepped_ind_bins_total,
         'Period': np.arange(len(stepped_ind_fees_total)),
         'Scenario': 'Stepped model'})

    ind_df_results = pd.concat([_ind_df_results, _ind_df_results2])
    return (ind_df_results,)


@app.cell
def _(
    coop_old_bin_ratio,
    coop_old_fees,
    coop_old_points,
    coop_setup_editor,
    np,
):
    def bin_count_response(price_increase: float, period:int, scale: float = 6.0):

        if period == 0:
            return 1

        return 1 - (price_increase / (scale ** period))

    def bin_per_point_forecast(initial_state: np.array, point_count: np.array, price_increase: float, periods: int):

        # for each period do this:
        ## get bin_count_response for given period and price increase
        ## multiply initial_state by this factor = new_state
        ## multiply point_count by new_state
        ## if there are any values < 1 in new_state, move the difference of 1 - new_state to lower frequency 
        ## adjust bins per point to 1 for those where it was < 1
        ## repeat 

        ## shape of initial_state = (1, n) where n is number of possible frequencies
        ## shape of point_count = (1, n) where n is number of possible frequencies
        ## 

        bin_count_history = []
        for period in range(periods):

            _factor = bin_count_response(price_increase, period)
            new_state = initial_state * _factor 
            below_one = new_state < 1

            if np.any(below_one):
                below_one_delta = np.maximum(0, 1 - new_state)
                point_shift = point_count * below_one_delta
                rolled_point_shift = np.roll(point_shift, -1)
                new_bins = (point_count * new_state) + rolled_point_shift
                point_count = point_count + rolled_point_shift - point_shift 
                new_state = new_bins / point_count
            else:
                new_bins = point_count * new_state # dimensions possibly need to be adjusted 

            if len(bin_count_history) == 0:
                bin_count_history = new_bins.T
            else:
                bin_count_history = np.concatenate([bin_count_history, new_bins.T])

            initial_state = new_state

        return bin_count_history, new_state


    def stepped_coop():

        old_fees = coop_old_fees.copy() 
        baseline_points = coop_old_points.copy() 
        baseline_ratio = coop_old_bin_ratio.copy()
        bin_evolution = np.array([])
        fee_evolution = np.array([])
        price_hike = 0.

        for step in coop_setup_editor.value.iterrows():
        
            step_values = step[1]
            price_hike = price_hike + (step_values['Fee Hike'] / 100.)
            _new_fees = old_fees * (1 + price_hike)
            _evolution, _latest_ratio = bin_per_point_forecast(baseline_ratio, baseline_points, price_hike, step_values['Year Gap']+1)
            _fee_evolution = _evolution * _new_fees.T 
            if len(bin_evolution) == 0:
                _fee_evolution[0] = _evolution[0] * old_fees.T
                bin_evolution = _evolution
                fee_evolution = _fee_evolution
            else:
                _fee_evolution = _fee_evolution[1:]
                _evolution = _evolution[1:]
                bin_evolution = np.concatenate([bin_evolution, _evolution])
                fee_evolution = np.concatenate([fee_evolution, _fee_evolution])
            
            baseline_bins = np.array([_evolution[-1]])
            baseline_points = (baseline_bins / _latest_ratio.T).T
            baseline_ratio = _latest_ratio

        return bin_evolution, fee_evolution

    return bin_per_point_forecast, stepped_coop


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Theory of change 

    Individual payers do not tend to change number of bins (becuase they almost always only have one and cannot not have trash collected), they alter their pickup frequency. 

    Coops and businesses can work with the number of bins as a more tangible way of saving. Reducing frequency from 3/week to 2/week saves 1/3 of costs, while removing one bin when you have two bins saves 1/2 of costs. This is most pronounced with business 120l cans, and to a lesser extent with standard 1100l coop cans. 

    So we will model impacts of fee/schedule changes this way:

    1. For individuals, we will create a transition matrix for frequencies (most common ones) as a function of fee change. Plus more custom matrices when we will model changes in available frequencies.
    2. For coops and businesses, we will do it this way:

    - Create a function that converts price hike into a factor for decreasing bins per collection point
    - We will assume that frequencies will not change unless the bins/collection point should drop below 1. In this case, we will move the overflow (or underflow) into a lower frequency.
    - So the bins per collection will be calculated on a per interval basis so that we can distribute them afterwards.
    """
    )
    return


@app.cell
def _(np):
    def matrix_model_individual(price_increase: float = 0.3, base_scale: float = 2.0, year: int = 1, remove_weekly: bool = False):

        """
        the matrix is:
        120l capacity, 1x month
        120l capacity, 2x month
        120l capacity, 4x month
        240l capacity, 2x month
        240l capacity, 4x month
        """

        factor1 = price_increase / 2 
        factor2 = price_increase / 15
        factor_split = (factor1 * 0.45, factor1 * 0.55)

        unit = lambda factor, year: factor / (base_scale ** year)

        monthly_unit = 1 - unit(factor1, year)
        monthly_split = unit(factor_split[0], year), unit(factor_split[1], year)
        big_monthly_unit = 1-unit(factor2, year)
        big_monthly_split = [0, 0, unit(factor2, year), 0, big_monthly_unit]
        if remove_weekly:
            monthly_unit = 0
            big_monthly_unit = 0
            monthly_split = 0.2, 0.8
            big_monthly_split = [0.05, 0.8, 0, 0.15, 0]

        matrix = [
            [1,0,0,0,0],
            [unit(factor2, year), 1-unit(factor2, year), 0, 0, 0],
            [monthly_split[0], monthly_split[1], monthly_unit, 0, 0],
            [0, unit(factor2, year), 0, 1-unit(factor2, year), 0],
            big_monthly_split
        ]

        return np.array(matrix)
    return (matrix_model_individual,)


@app.cell
def _(matrix_model_individual, np):
    def run_individual(initial_state: np.array, years: int = 10, current_year: int = 0, **kwargs):

        price_increase = kwargs.get('price_increase',0.3)
        base_scale = kwargs.get('base_scale',2)
        remove_weekly = kwargs.get('remove_weekly', False)

        if current_year < years:
            matrix = matrix_model_individual(price_increase, base_scale, current_year+1, remove_weekly)
            step = np.matmul(matrix.T, np.array([initial_state[-1]]).T)
            history = np.concatenate([initial_state, step.T])
            return run_individual(history, years, current_year+1, **kwargs)

        return initial_state
    return (run_individual,)


if __name__ == "__main__":
    app.run()
