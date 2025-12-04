import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import micropip
    return (micropip,)


@app.cell
async def _(micropip):
    await micropip.install("altair")
    await micropip.install("openpyxl")
    return


@app.cell
def _():
    import marimo as mo 
    import pandas as pd 
    import numpy as np
    import altair as alt
    import os 
    return alt, mo, np, pd


@app.cell
def _(mo, pd):
    fee_data = pd.read_excel(str(mo.notebook_location() / 'public' / 'fee_data.xlsx'))
    olo_data = pd.read_excel(str(mo.notebook_location() / 'public' / 'olo_exp.xlsx'))
    yard_data = pd.read_excel(str(mo.notebook_location() / 'public' / 'yards.xlsx'))
    return fee_data, olo_data, yard_data


@app.cell
def _(np, olo_data, pd):
    # clean OLO data
    transposed_olo = olo_data.T
    transposed_olo.columns = transposed_olo.iloc[0]
    OLO_base = transposed_olo.drop(transposed_olo.index[0]).reset_index()[['index','NakladyOLO']].rename(columns={'index':'year', 'NakladyOLO':'Cost of OLO'})

    # VAT
    VAT = 0.23
    OLO_base['Cost of OLO'] = OLO_base['Cost of OLO'] * (1 + VAT)

    # extend expenses further
    _olo_years = OLO_base['year'].count() - 1
    _olo_min = OLO_base['Cost of OLO'].values[0]
    _olo_max = OLO_base['Cost of OLO'].values[-1]
    olo_rate_of_growth = np.power((_olo_max / _olo_min), 1/_olo_years)
    _last_year = OLO_base['year'].max()
    end_of_forecast = 2041
    for y in range(_last_year + 1, end_of_forecast):
        _new_row = {
            'year': y, 
            'Cost of OLO': _olo_max * (olo_rate_of_growth ** (y - _last_year)) 
        }
        OLO_base = pd.concat([OLO_base, pd.DataFrame(_new_row, index=[0])], ignore_index=True)

    OLO_base['Cost of OLO'] = OLO_base['Cost of OLO'].astype('float64')
    #
    return OLO_base, end_of_forecast, olo_rate_of_growth


@app.cell
def _(end_of_forecast, olo_rate_of_growth, pd, yard_data):
    yard_data['year'] = 2024
    yard_pivot = yard_data.pivot(columns=['Mestska_cast'],index=['year'],values=['Naklady_zhodnotenie'])
    yard_total = yard_pivot.sum(axis=1).reset_index()
    yard_total.columns = ['year','yard_cost']
    _cost = yard_total.yard_cost.values[0]
    for _y in range(2025, end_of_forecast):
        _new_row = {
            'year': _y, 
            'yard_cost': _cost * (olo_rate_of_growth ** (_y - 2024))
        }
        yard_total = pd.concat([yard_total, pd.DataFrame(_new_row, index=[0])], ignore_index=True)

    yard_total['Period'] = yard_total['year'] - 2025
    return (yard_total,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Simulation of PAYT reform

    In this notebook we can simulate impacts of waste collection reforms on tax revenue and expenditures of OLO. 
    We will adjust different parameters to show the expected evolution of key indicators in various scenarios. Since we have little available past data on how people in Bratislava respond to changes in waste collection, we need to model different scenarios to show a potential range of outcomes. 

    There are general assumptions you can setup: 

    1. Annual rate of growth for fee collection. Default is `1.5%` per year. This represents new construction and more people moving in to the city.
    2. Share of individual bins that are filled to capacity. Default is `53%`. This reflects latest results from a study of individual homes. This impacts how many people move to larger bins.
    3. Assumption for how much more expensive OLO is compared to local yards. Default is `1.5x`. This reflects that when local districts handle waste, they usually choose cheaper options.
    4. Whether OLO should take over local waste collection yards. If yes, the collection costs are added to costs of OLO.
    5. Behavioral scenario (see below). Total results show results for all scenarios.
    6. Baseline value for other expenses covered by the waste fees. 

    Here you can set up the simulation: 

    1. Increase in fees (in %)
    2. Removing / adding options of collection schedules
    3. Number of people per large 1,100L bin

    Run the model by pressing the green button **'Run Model'**.

    You can also adjust the assumed cost of OLO to the city by pressing the yellow **'Edit/Reset OLO costs'**. A table will appear where you can edit costs for each year until year `2040`. Upon changing the table, results will be automatically updated. Pressing the yellow button again will reset the values to initital assumptions.

    The simulation is parametrized differently for individual homes and differently for businesses/coops. This is because the assumption is that these groups respond differently to fee increases. While individual home owners can respond by frequency changes only (usually only have on bin), coops and businesses mostly adjust the number of bins, and only if this is not an option do they drop frequencies. 

    You can setup a simple model (one % fee increase) or a stepped model where we can schedule several adjustments to fees over time.

    ## Sensitivity settings (behavioral response)

    The default model assumes a behavioral response to the changes in fees and other conditions that is most consistent with past trends. This data is very limited, and so we need to model different scenarios of how people might react to policy changes. 

    You can choose these following scenarios to simulate:

    1. **No behavioral change**: This scenario assumes no behavioral changes to an increase in fees. The only reaction that is required is when the weekly schedule is disabled. This will lead to a transfer of all households with a weekly schedule to a fortnightly schedule with the larger bin.
    2. **Standard response**: This is the default scenario that mostly follows an extrapolation of what was observed after the last change to the fee structure. It assumes a fairly muted behavioral response to an increase in fees. However, the capacity of households to bear higher fees is assumed to fall with greater increases (even when you stagger them in multiple steps).
    3. **Strong response**: This is a scenario where we assume that compared to 2023, the financial situation of households has worsened (following inflation and increases in other national taxes). Therefore the reaction to a higher waste fee will be more pronounced this time.

    In the results, we also show a scenario of **No Policy Change**. That is, a scenario without changes to fees or other reforms that would affect revenues. 
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # parametrize rate of growth for fees BAU, waste collection yards...

    global_fee_growth = mo.ui.slider(0, 10, 0.1, include_input=True, show_value=True, value=1.5)
    yard_takeover = mo.ui.checkbox(value=False)
    average_bin_fullness = mo.ui.slider(0,100,show_value=True, include_input=True, value=53)
    scenario_selector = mo.ui.dropdown(options = {'1 No Change': 3, '2 Standard Response': 1, '3 Strong Response': 2},
                                       allow_select_none=False,
                                       searchable=True, 
                                       value='2 Standard Response'
                                      )

    other_cost_baseline = mo.ui.number(start=0, value=1070384)
    olo_multiplier = mo.ui.slider(0,10,0.1, include_input=True, show_value=True, value=1.5)

    mo.vstack([
        mo.md('## Parameter setup'),
        mo.md('### General assumptions and policy changes'),
        mo.hstack([
            mo.vstack([
                mo.md('**Annual rate of growth for fee collection (new construction, etc.) in %**'),
                global_fee_growth
            ]),
            mo.vstack([
                mo.md('**Make OLO handle waste collection from local waste collection yards**'), 
                yard_takeover
            ])]),
        mo.hstack([
            mo.vstack([
                mo.md('**Share of individual bins that are filled to capacity in %**'), 
                average_bin_fullness
            ]),
            mo.vstack([
                mo.md('**Behavioral scenario**'),
                scenario_selector
            ])
        ]),
        mo.hstack([
            mo.vstack([
                mo.md('**Assumption for how much more expensive OLO is compared to local yards (multiplier)**'),
                olo_multiplier
            ]),
            mo.vstack([
                mo.md('**2024 value for other expenses covered by waste fees**'), 
                other_cost_baseline
            ])
        ])
    ])
    return (
        average_bin_fullness,
        global_fee_growth,
        olo_multiplier,
        other_cost_baseline,
        scenario_selector,
        yard_takeover,
    )


@app.cell(hide_code=True)
def _(mo, pd):
    fee_hike = mo.ui.slider(0,100, show_value=True, include_input=True, value=30)
    remove_weekly = mo.ui.checkbox()
    forecast_periods = mo.ui.slider(0,20, show_value=True, include_input=True, value=5)

    ind_setup_df = pd.DataFrame(columns=['Fee Hike','Year Gap','Maximum 14d Interval'], data=[[30,5,False]])
    ind_setup_editor = mo.ui.data_editor(data=ind_setup_df)


    mo.vstack([
        mo.md('### Individual homes (simple setup)'),
        mo.hstack([mo.md('**1 Increase in standard fee in %**'), fee_hike], align='center'),
        mo.hstack([mo.md('**2 Force a maximum 14d interval for individual homes**'), remove_weekly],  align='center'),
        mo.hstack([mo.md('**3 Number of years to forecast**'), forecast_periods], align='center'),
        mo.md('### Individual homes (complex setup)'),
        mo.md('In this table, you can add multiple rows as steps in gradual fee increase / change'),
        ind_setup_editor

    ])
    return fee_hike, forecast_periods, ind_setup_editor, remove_weekly


@app.cell(hide_code=True)
def _(mo, pd):

    fee_hike_coop = mo.ui.slider(0,100, show_value=True, include_input=True, value=30)
    persons_per_bin_coop = mo.ui.slider(45,200,show_value=True, include_input=True, value=45)
    forecast_periods_coop = mo.ui.slider(0,20, show_value=True, include_input=True, value=5)

    coop_setup_df = pd.DataFrame(columns=['Fee Hike','Number of people per bin','Year Gap'], data=[[30,45,5]])
    coop_setup_editor = mo.ui.data_editor(data=coop_setup_df)

    mo.vstack([
         mo.md('### Coops/businesses (simple setup)'),
        mo.hstack([mo.md('**1 Increase in standard fee in %**'), fee_hike_coop], align='center'),
        mo.hstack([mo.md('**2 Number of people per 1110l bin**'), persons_per_bin_coop], align='center'),
        mo.hstack([mo.md('**3 Number of years to forecast**'), forecast_periods_coop], align='center'),
        mo.md('### Coops/businesses (complex setup)'),
        coop_setup_editor
    ])
    return (
        coop_setup_editor,
        fee_hike_coop,
        forecast_periods_coop,
        persons_per_bin_coop,
    )


@app.cell
def _(mo):
    olo_edit_button = mo.ui.run_button(label='Edit/Reset OLO costs', kind='warn')
    general_run_button = mo.ui.run_button(label='Run Model', kind='success')

    mo.hstack([general_run_button, olo_edit_button])
    return general_run_button, olo_edit_button


@app.cell(hide_code=True)
def _(alt, mo, results_overview):
    results_together, detailed_results_together = results_overview()

    def summary_chart(model, result_df):

        _df = result_df.query('Model == @model')[['Period','Fee_forecast','Scenario']].copy()
        _exp_df = result_df.query('Model == @model and Scenario == "2 Standard Response"').drop(columns=['Fee_forecast','Bin_forecast','Scenario'])
        _melt_exp = _exp_df.melt(id_vars=['Period'], value_vars=['1 OLO', '2 Other Expenses', '3 MC Share', '4 OLO Yards'], var_name='Category',value_name='Expenses').sort_values(by=['Category'], ascending=True)

        lines_chart = alt.Chart(_df).mark_trail().encode(alt.X('Period').axis(format='0d', 
                                                                             tickCount=_df.Period.max()), 
                                                         y='Fee_forecast',
                                            color=alt.Color('Scenario').scale(scheme='paired'))

        bar_chart = alt.Chart(_melt_exp).mark_bar(size=25).encode(alt.X('Period').axis(format='0d', 
                                                                             tickCount=_df.Period.max()),y='Expenses',color='Category')

        return lines_chart + bar_chart 


    total_simple = mo.ui.altair_chart(summary_chart('Simple model', results_together), label="Total fee forecast per behavioral scenario (Simple)")
    total_stepped = mo.ui.altair_chart(summary_chart('Stepped model', results_together), label="Total fee forecast per behavioral scenario (Stepped)")

    mo.vstack([
        mo.md('## Total results for all behavioral scenarios'),
        total_simple,
        total_stepped
    ])
    return (results_together,)


@app.cell
def _(results_together):
    results_together
    return


@app.cell
def _(OLO_base, general_run_button, mo, olo_edit_button):
    mo.stop(not (olo_edit_button.value or general_run_button.value), mo.md('***Run the model to show full results***'))
    olo_editor = mo.ui.data_editor(data=OLO_base)

    mo.vstack([
        mo.md('### Adjust expected cost of OLO'), 
        mo.md('You can adjust the expected cost to the city of the service of OLO in EUR per year, incl. VAT of 23 %:'), 
        olo_editor
    ])
    return (olo_editor,)


@app.cell
def _(olo_editor):
    OLO = olo_editor.value
    OLO['Period'] = OLO['year'] - 2025
    OLO.rename(columns={'Cost of OLO': '1 OLO'}, inplace=True)
    return (OLO,)


@app.cell(hide_code=True)
def _(scenario_selector):
    SCENARIO = scenario_selector.value
    return (SCENARIO,)


@app.cell(hide_code=True)
def _(coop_results_, ind_df_results, pd):
    results_summary = ind_df_results.query('Period < 2').copy().rename(columns={'Individual payers fees': 'Fees', 'Individual bin count': 'Bins'})
    results_summary['Payer'] = 'Individual'

    _coop_period1 = coop_results_.query('Period < 2').copy().rename(columns={'Coop fees': 'Fees', 'Coop bin count': 'Bins'})
    _coop_period1['Payer'] = 'Coop'

    results_summary = pd.concat([results_summary, _coop_period1], ignore_index=True)
    results_summary["Period"] = results_summary["Period"].astype(str).str.replace('0','0: Before Change').str.replace('1','1: First year after change')
    return (results_summary,)


@app.cell(hide_code=True)
def _(coop_results_, ind_df_results, mo, pd, results_summary):
    last_period_ind = ind_df_results.groupby(['Scenario'], as_index=False)['Period'].max()
    last_period_coop = coop_results_.groupby(['Scenario'], as_index=False)['Period'].max()

    latest_ind = ind_df_results.merge(last_period_ind, on=["Scenario",'Period'], how='inner').rename(columns={'Individual payers fees': 'Fees', 'Individual bin count': 'Bins'})
    latest_ind['Period'] = '2: Last forecasted period'
    latest_ind['Payer'] = 'Individual'
    latest_coop = coop_results_.merge(last_period_coop, on=["Scenario",'Period'], how='inner').rename(columns={'Coop fees': 'Fees', 'Coop bin count': 'Bins'})
    latest_coop['Period'] = '2: Last forecasted period'
    latest_coop['Payer'] = 'Coop'

    results_summary_latest = pd.concat([results_summary, latest_ind, latest_coop], ignore_index=True)
    results_summary_latest['Fees'] = results_summary_latest['Fees'].round(0).astype(int)
    results_summary_totals = results_summary_latest.groupby(['Scenario','Period'], as_index=False)['Fees'].sum()
    results_summary_totals['Payer'] = 'Total'

    final_results = pd.concat([results_summary_latest, results_summary_totals], ignore_index=True).sort_values(by=['Scenario', 'Payer', 'Period'])

    results_pivot = pd.pivot(final_results, columns='Period', index=['Scenario','Payer'], values=['Fees'])

    mo.vstack([
        mo.md('## Simulation results (Fees)'),
        mo.md('Below you can see the total results for the current simulation setup for fees in EUR.'),
        mo.md('The table shows fees before change implementation, first year after implementation and in the last projected period.'),
        mo.md('Then the following charts show a detailed evolution over time of fee collection and the number of bins.'),
        results_pivot
    ])
    return


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
def _(fee_data):
    # get baseline for coops and businesses homes
    coop_sample = fee_data.query('Payer != "Individual"')
    coop_baseline = coop_sample.query('Year == 2025').groupby(['IntervalPerWeek'], as_index=False)[['CollectionPoints','BinCount','TotalFee','TotalVolume']].sum()
    coop_baseline['FeePerBin'] = coop_baseline['TotalFee'].div(coop_baseline['BinCount'])
    coop_baseline['BinPerPoint'] = coop_baseline['BinCount'].div(coop_baseline['CollectionPoints'])

    coop_old_fees = coop_baseline[['FeePerBin']].values
    coop_old_bin_ratio = coop_baseline[['BinPerPoint']].values
    coop_old_points = coop_baseline[['CollectionPoints']].values
    return coop_old_bin_ratio, coop_old_fees, coop_old_points


@app.cell
def _(fee_data):
    baseline_fees = fee_data.query('Year == 2025').groupby(['Year'], as_index=False)[['CollectionPoints','BinCount','TotalFee']].sum()
    return (baseline_fees,)


@app.cell
def _(global_fee_growth, np):
    def indexer(periods: np.array):
        rate_of_inc = global_fee_growth.value / 100 + 1
        rate_array = np.full(shape=periods.shape, fill_value=rate_of_inc)
        index_array = rate_array ** periods
        return index_array
    return (indexer,)


@app.cell
def _(
    SCENARIO,
    bin_per_point_forecast,
    coop_old_bin_ratio,
    coop_old_fees,
    coop_old_points,
    fee_hike_coop,
    forecast_periods_coop,
    indexer,
    np,
    pd,
    persons_per_bin_coop,
):
    def coop_result_simple(initial_state: np.array, initial_fees: np.array, point_count: np.array, price_increase: float, periods: int, people_per_bin: int = 45, scenario: int = 1):

        # define values
        coop_fee_hike_pct = price_increase/100.0
        coop_new_fees = initial_fees * (1 + coop_fee_hike_pct)

        simple_coop_forecast_bins, latest_sample = bin_per_point_forecast(initial_state, point_count, coop_fee_hike_pct, periods, people_per_bin=people_per_bin, scenario=scenario)
        simple_coop_forecast_fees = simple_coop_forecast_bins * coop_new_fees.T
        simple_coop_forecast_fees[0] = simple_coop_forecast_bins[0] * initial_fees.T 

        periods = np.arange(periods)
        index_array = indexer(periods)
        fees = simple_coop_forecast_fees.sum(axis=1) * index_array

        results = pd.DataFrame({'Coop fees': fees,
                               'Coop bin count': simple_coop_forecast_bins.sum(axis=1),
                               'Period': periods,
                                'Scenario': 'Simple model'})
        return results 

    coop_results = coop_result_simple(coop_old_bin_ratio, coop_old_fees, coop_old_points, fee_hike_coop.value, forecast_periods_coop.value, people_per_bin=persons_per_bin_coop.value, scenario=SCENARIO)
    return coop_result_simple, coop_results


@app.cell
def _(SCENARIO, coop_results, indexer, np, pd, stepped_coop):
    def coop_result_stepped(scenario: int = 1):

        stepped_coop_bins, stepped_coop_fees = stepped_coop(scenario)
        stepped_coop_bins_total = stepped_coop_bins.sum(axis=1)
        stepped_coop_fees_total = stepped_coop_fees.sum(axis=1)

        periods = np.arange(len(stepped_coop_fees_total))
        index_array = indexer(periods)
        stepped_coop_fees_total = stepped_coop_fees_total * index_array


        return pd.DataFrame(
            {'Coop fees': stepped_coop_fees_total,
             'Coop bin count': stepped_coop_bins_total,
             'Period': periods,
             'Scenario': 'Stepped model'})

    coop_results2 = coop_result_stepped(SCENARIO)

    coop_results_ = pd.concat([coop_results, coop_results2], ignore_index=True)
    return coop_result_stepped, coop_results_


@app.cell
def _(fee_data):
    # get baseline for individual homes
    individual_sample = fee_data.query('Payer == "Individual" and IntervalPerWeek < 2 and CapacityInt < 1000')
    ind_baseline = individual_sample.query('Year == 2025')[['CapacityInt','IntervalPerWeek','CollectionPoints','BinCount','TotalFee','TotalVolume']]
    ind_baseline['FeePerBin'] = ind_baseline['TotalFee'].div(ind_baseline['BinCount'])
    return (ind_baseline,)


@app.cell
def _(ind_baseline):
    # get baseline for bins and fees for homes
    ind_old_fees = ind_baseline[['FeePerBin']].values 
    ind_baseline_bins = ind_baseline[['BinCount']].values.T
    return ind_baseline_bins, ind_old_fees


@app.cell
def _(ind_old_fees, indexer, np, pd, run_individual):
    # forecast individual homes
    def ind_results_simple(initial_state: np.array, initial_fees: np.array, periods: int, price_increase: float, remove_weekly: bool, scenario: int):

        # define values
        ind_fee_hike_pct = price_increase/100.0
        ind_new_fees = ind_old_fees * (1. + ind_fee_hike_pct)

        # compute 
        ind_bin_evolution = run_individual(initial_state, periods, 0, price_increase=ind_fee_hike_pct, remove_weekly=remove_weekly, scenario=scenario)
        ind_fee_evolution = ind_bin_evolution * ind_new_fees.T
        ind_fee_evolution[0] = ind_bin_evolution[0] * initial_fees.T # first year keeps old fees

        ind_fee_evolution_total = ind_fee_evolution.sum(axis=1)
        ind_bin_evolution_total = ind_bin_evolution.sum(axis=1)

        periods = np.arange(periods)
        index_array = indexer(periods)
        ind_fee_evolution_total = ind_fee_evolution_total * index_array

        return pd.DataFrame(
        {'Individual payers fees': ind_fee_evolution_total,
         'Individual bin count': ind_bin_evolution_total,
         'Period': periods,
         'Scenario': 'Simple model'})
    return (ind_results_simple,)


@app.cell
def _(
    SCENARIO,
    ind_baseline_bins,
    ind_old_fees,
    ind_setup_editor,
    indexer,
    np,
    pd,
    run_individual,
):
    def stepped_individual(scenario:int = 1):

        old_fees = ind_old_fees.copy() 
        baseline_bins = ind_baseline_bins.copy() 
        bin_evolution = np.array([])
        fee_evolution = np.array([])
        price_hike = 0.

        for step in ind_setup_editor.value.iterrows():
            step_values = step[1]
            price_hike = price_hike + (step_values['Fee Hike'] / 100.)
            _new_fees = old_fees * (1 + price_hike)
            _evolution = run_individual(baseline_bins, step_values['Year Gap']+1, 0, price_increase = price_hike, remove_weekly = step_values['Maximum 14d Interval'], scenario = scenario)
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

    def ind_results_stepped(scenario: int): 

        stepped_ind_bins, stepped_ind_fees = stepped_individual(SCENARIO)
        stepped_ind_bins_total = stepped_ind_bins.sum(axis=1)
        stepped_ind_fees_total = stepped_ind_fees.sum(axis=1)

        periods = np.arange(len(stepped_ind_fees_total))
        index_array = indexer(periods)
        stepped_ind_fees_total = stepped_ind_fees_total * index_array

        return pd.DataFrame(
        {'Individual payers fees': stepped_ind_fees_total,
         'Individual bin count': stepped_ind_bins_total,
         'Period': periods,
         'Scenario': 'Stepped model'})
    return (ind_results_stepped,)


@app.cell
def _(
    SCENARIO,
    fee_hike,
    forecast_periods,
    ind_baseline_bins,
    ind_old_fees,
    ind_results_simple,
    ind_results_stepped,
    pd,
    remove_weekly,
):
    _ind_df_results = ind_results_simple(ind_baseline_bins, ind_old_fees, forecast_periods.value, fee_hike.value, remove_weekly.value, SCENARIO)
    _ind_df_results2 = ind_results_stepped(SCENARIO)

    ind_df_results = pd.concat([_ind_df_results, _ind_df_results2])
    return (ind_df_results,)


@app.cell
def _(
    OLO,
    baseline_fees,
    coop_old_bin_ratio,
    coop_old_fees,
    coop_old_points,
    coop_result_simple,
    coop_result_stepped,
    fee_hike,
    fee_hike_coop,
    forecast_periods,
    forecast_periods_coop,
    ind_baseline_bins,
    ind_old_fees,
    ind_results_simple,
    ind_results_stepped,
    indexer,
    np,
    olo_multiplier,
    other_cost_baseline,
    pd,
    persons_per_bin_coop,
    remove_weekly,
    yard_takeover,
    yard_total,
):
    def equalize_frames(longer_frame: pd.DataFrame, shorter_frame: pd.DataFrame, col_name: str = 'Period'):
        minmax = shorter_frame.Period.max()
        delta = longer_frame.Period.max() - minmax
        last_row = shorter_frame.query(f'{col_name} == @minmax').copy()
        for step in range(1, delta+1):
            last_row[col_name] = minmax + step 
            shorter_frame = pd.concat([shorter_frame, last_row])

        return shorter_frame

    def build_expenses():

        expenses = OLO.copy()
      
        expenses['4 OLO Yards'] = 0.

        if yard_takeover.value:
            expenses = expenses.merge(yard_total[['year','yard_cost']], on='year', how='left')
            expenses['4 OLO Yards'] = expenses['yard_cost'] * olo_multiplier.value
            expenses.drop(columns=['yard_cost'], inplace=True)

        other_exp_base = other_cost_baseline.value 
        periods = expenses['Period'].values 
        index = indexer(periods)
        other_exp = index * other_exp_base

        expenses['2 Other Expenses'] = other_exp 
        return expenses

    def results_overview():

        """
        We are going to generate results for all behavioral scenarios 
        for both individual and coop results. 

        Then we aggregate ind + coop for each scenario and each model option (simple/stepped) per period.

        helpers

        coop_result_simple(coop_old_bin_ratio, coop_old_fees, coop_old_points, fee_hike_coop.value, forecast_periods_coop.value, people_per_bin=persons_per_bin_coop.value, scenario=SCENARIO)

        coop_result_stepped(SCENARIO)

        ind_results_simple(ind_baseline_bins, ind_old_fees, forecast_periods.value, fee_hike.value, remove_weekly.value, SCENARIO)

        ind_results_stepped(SCENARIO)
        """
        simple_detail = []
        stepped_detail = []
        simple_results = []
        stepped_results = []
        COLS = ['Fee_forecast','Bin_forecast','Period','Model']
        scenarios = {
            1: '2 Standard Response',
            2: '3 Strong Response',
            3: '1 No Behavioral Change'
        }

        periods = [0,0]

        for scenario in [1,2,3]:

            coop_simple = coop_result_simple(coop_old_bin_ratio, coop_old_fees, coop_old_points, fee_hike_coop.value, 
                                            forecast_periods_coop.value, people_per_bin=persons_per_bin_coop.value, scenario=scenario)
            coop_simple.columns = COLS

            coop_stepped = coop_result_stepped(scenario)
            coop_stepped.columns = COLS

            ind_simple = ind_results_simple(ind_baseline_bins, ind_old_fees, forecast_periods.value, fee_hike.value, remove_weekly.value, scenario)
            ind_simple.columns = COLS

            ind_stepped = ind_results_stepped(scenario)
            ind_stepped.columns = COLS

            # check need to extend forecast in one or other result
            if ind_simple.shape[0] > coop_simple.shape[0]:
                coop_simple = equalize_frames(ind_simple, coop_simple)
            elif ind_simple.shape[0] < coop_simple.shape[0]:
                ind_simple = equalize_frames(coop_simple, ind_simple)

            periods[0] = ind_simple.shape[0]

            # check need to extend forecast in one or other result
            if ind_stepped.shape[0] > coop_stepped.shape[0]:
                coop_stepped = equalize_frames(ind_stepped, coop_stepped)
            elif ind_stepped.shape[0] < coop_stepped.shape[0]:
                ind_stepped = equalize_frames(coop_stepped, ind_stepped)

            periods[1] = ind_stepped.shape[0]

            simple = pd.concat([coop_simple, ind_simple], ignore_index=True)
            simple['Scenario'] = scenarios[scenario]
            simple_detail.append(simple)
            simple_group = simple.groupby(['Scenario','Model','Period'], as_index=False)[['Fee_forecast','Bin_forecast']].sum()
            simple_results.append(simple_group)

            stepped = pd.concat([coop_stepped, ind_stepped], ignore_index=True)
            stepped['Scenario'] = scenarios[scenario]
            stepped_detail.append(stepped)
            stepped_group = stepped.groupby(['Scenario','Model','Period'], as_index=False)[['Fee_forecast','Bin_forecast']].sum()
            stepped_results.append(stepped_group)

        # add NPC scenario 

        simple_indexer = indexer(np.arange(periods[0]))
        stepped_indexer = indexer(np.arange(periods[1]))
        start_fee = baseline_fees['TotalFee'].values[0]
        simple_npc_fees = start_fee * simple_indexer 
        stepped_npc_fees = start_fee * stepped_indexer 

        simple_npc_df = pd.DataFrame(data={
            'Scenario': ['0 No Policy Change']*periods[0],
            'Model': ['Simple model']*periods[0],
            'Period': np.arange(periods[0]),
            'Fee_forecast': simple_npc_fees}, index=np.arange(periods[0]))

        stepped_npc_df = pd.DataFrame(data={
            'Scenario': ['0 No Policy Change']*periods[1],
            'Model': ['Stepped model']*periods[1],
            'Period': np.arange(periods[1]),
            'Fee_forecast': stepped_npc_fees}, index=np.arange(periods[1]))
    

        together = pd.concat(simple_results+stepped_results+[simple_npc_df, stepped_npc_df], ignore_index=True)
        together_detail = pd.concat(simple_detail+stepped_detail, ignore_index=True)

        # merge OLO
        expenses = build_expenses()
        together = together.merge(expenses, on=['Period'], how='left')
        together['3 MC Share'] = together.Fee_forecast * 0.1

        return together, together_detail


    return (results_overview,)


@app.cell
def _(
    coop_old_bin_ratio,
    coop_old_fees,
    coop_old_points,
    coop_setup_editor,
    np,
):
    def bin_count_response(price_increase: float, period:int, people_per_bin:int = 45, previous_per_bin:int = 45, scale: float = 6.0, scenario: int = 1):

        if period == 0 or scenario == 3:
            return 1

        if scenario == 2:
            scale = 3.5

        per_bin_factor = 1
        if previous_per_bin < people_per_bin:
            per_bin_factor = 1 - (np.log10(people_per_bin - 44) / (scale * 3.5))

        return (1 - (price_increase / (scale ** period)))*per_bin_factor

    def bin_per_point_forecast(initial_state: np.array, point_count: np.array, price_increase: float, periods: int, people_per_bin: int = 45, scenario: int = 1):

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
        previous_per_bin = 45
        for period in range(periods):

            _factor = bin_count_response(price_increase, period, people_per_bin, previous_per_bin, scenario=scenario)
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

            if period > 0:
                previous_per_bin = people_per_bin
            initial_state = new_state

        return bin_count_history, new_state


    def stepped_coop(scenario: int = 1):

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
            _evolution, _latest_ratio = bin_per_point_forecast(baseline_ratio, baseline_points, price_hike, step_values['Year Gap']+1, step_values['Number of people per bin'], scenario=scenario)
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


@app.cell(hide_code=True)
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
def _(average_bin_fullness, np):
    def matrix_model_individual(price_increase: float = 0.3, base_scale: float = 2.0, year: int = 1, remove_weekly: bool = False, scenario: int = 1):

        """
        the matrix is:
        120l capacity, 1x month
        120l capacity, 2x month
        120l capacity, 4x month
        240l capacity, 2x month
        240l capacity, 4x month
        """

        if scenario == 3:
            if not remove_weekly:
                return np.identity(5)

            _matrix = np.identity(5)
            _matrix[2,:] = _matrix[3,:]
            _matrix[4,:] = _matrix[3,:]
            return _matrix

        if scenario == 2:
            base_scale = 1.5

        factor1 = price_increase / 2 
        factor2 = price_increase / 15
        factor_split = (factor1 * 0.45, factor1 * 0.55)

        unit = lambda factor, year: factor / (base_scale ** year)

        monthly_unit = 1 - unit(factor1, year)
        monthly_split = [unit(factor_split[0], year), unit(factor_split[1], year), monthly_unit, 0 , 0]
        big_monthly_split = [0, 0, monthly_split[0], monthly_split[1], monthly_unit]
        if remove_weekly:
            monthly_unit = 0
            full_share = average_bin_fullness.value / 100
            monthly_split = [0.2*(1-full_share), 0.8*(1-full_share), 0, full_share, 0]
            big_monthly_split = [0.05, 0.15, 0, 0.8, 0]




        matrix = [
            [1,0,0,0,0],
            [unit(factor2, year), 1-unit(factor2, year), 0, 0, 0],
            monthly_split,
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
        scenario = kwargs.get('scenario', 1)

        if current_year < years-1:
            matrix = matrix_model_individual(price_increase, base_scale, current_year+1, remove_weekly, scenario=scenario)
            step = np.matmul(matrix.T, np.array([initial_state[-1]]).T)
            history = np.concatenate([initial_state, step.T])
            return run_individual(history, years, current_year+1, **kwargs)

        return initial_state
    return (run_individual,)


if __name__ == "__main__":
    app.run()
