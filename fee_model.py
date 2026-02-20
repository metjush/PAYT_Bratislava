import marimo

__generated_with = "0.19.11"
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
    import io

    return alt, io, mo, np, pd


@app.cell
def _(io, mo, olo_upload, pd):
    fee_data = pd.read_excel(str(mo.notebook_location() / 'public' / 'fee_data.xlsx'))
    if len(olo_upload.value) == 0:
        olo_data = pd.read_excel(str(mo.notebook_location() / 'public' / 'olo_exp.xlsx'))
    else:
        olo_data = pd.read_excel(io.BytesIO(olo_upload.value[0].contents))
    yard_data = pd.read_excel(str(mo.notebook_location() / 'public' / 'yards.xlsx'))
    return fee_data, olo_data, yard_data


@app.cell
def _(np, olo_data, pd):
    # clean OLO data
    transposed_olo = olo_data.set_index('Rok').T.drop(columns=['NakladyOLO'])
    ## VAT
    VAT = 0.23
    transposed_olo = transposed_olo * (1 + VAT)


    OLO_base = transposed_olo.reset_index().rename(columns={'index':'year'})

    # extend expenses further
    _olo_years = OLO_base['year'].count() - 1
    _last_year = OLO_base['year'].max()
    end_of_forecast = 2041

    forecasted_olos = []

    for y in range(_last_year + 1, end_of_forecast):
        _new_row = {
            'year': y
        }
        for col in OLO_base.columns:
            if col == 'year':
                continue 
            _olo_min = OLO_base[col].values[0]
            _olo_max = OLO_base[col].values[-1]
            _olo_rate_of_growth = np.power((_olo_max / _olo_min), 1/_olo_years)
            _new_row[col] = _olo_max * _olo_rate_of_growth 

        forecasted_olos.append(pd.DataFrame(_new_row, index=[0]))

    OLO_base = pd.concat([OLO_base] + forecasted_olos , ignore_index=True)
    return OLO_base, end_of_forecast


@app.cell
def _(OLO_base, end_of_forecast, np, pd, yard_data, yard_takeover):
    yard_data['year'] = 2024
    yard_pivot = yard_data.pivot(columns=['Mestska_cast'],index=['year'],values='Naklady_zhodnotenie')
    _cols = yard_pivot.columns.values 
    if yard_takeover.value == 1:
        _cols = ['DevinskaNovaVes','Petrzalka','ZahorskaBystrica','Lamac']
    yard_total = yard_pivot[_cols].sum(axis=1).reset_index()
    yard_total.columns = ['year','yard_cost']
    olo_rate_of_growth = np.power(OLO_base.sum(axis=1).max() / OLO_base.sum(axis=1).min(), 1/OLO_base.shape[0])
    _cost = yard_total.yard_cost.values[0]
    for _y in range(2025, end_of_forecast):
        _new_row = {
            'year': _y, 
            'yard_cost': _cost * (olo_rate_of_growth ** (_y - 2024))
        }
        yard_total = pd.concat([yard_total, pd.DataFrame(_new_row, index=[0])], ignore_index=True)

    yard_total['Rok'] = yard_total['year'] - 2025
    return (yard_total,)


@app.cell(hide_code=True)
def _(mo):
    language_switch = mo.ui.switch(value=False, label='Translate to English')
    language_switch
    return (language_switch,)


@app.cell
def _(language_switch, mo, pd):
    translation_list = pd.read_excel(str(mo.notebook_location() / 'public' / 'translation.xlsx'))
    def translation(key: str):
        row = translation_list.query('key == @key')
        if language_switch.value:
            return row['english'].values[0]
        return row['slovak'].values[0]

    return (translation,)


@app.cell(hide_code=True)
def _(mo, translation):
    s = mo.md(translation('intro'))
    s
    return


@app.cell
def _(mo, translation):
    # parametrize rate of growth for fees BAU, waste collection yards...

    global_fee_growth = mo.ui.slider(0, 10, 0.1, include_input=True, show_value=True, value=1.5)
    yard_takeover = mo.ui.dropdown(options = {translation('d1'): 0, translation('d2'): 1, translation('d3'): 2}, allow_select_none=False, searchable=True, value=translation('d1'))
    average_bin_fullness = mo.ui.slider(0,100,show_value=True, include_input=True, value=53)
    scenario_selector = mo.ui.dropdown(options = {translation('d4'): 3, translation('d5'): 1, translation('d6'): 2},
                                       allow_select_none=False,
                                       searchable=True, 
                                       value=translation('d5')
                                      )

    other_cost_baseline = mo.ui.number(start=0, value=1070384)
    olo_multiplier = mo.ui.slider(0,10,0.1, include_input=True, show_value=True, value=1.5)
    organic_waste_pickup = mo.ui.dropdown(options = {translation('d7'): 0, translation('d8'): 1, translation('d9'): 2},
                                          allow_select_none=False,
                                         value=translation('d8'))

    olo_upload = mo.ui.file(filetypes=['.xls','.xlsx'], multiple=False, kind='button')

    mo.vstack([
        mo.md(translation('p1')),
        mo.md(translation('p2')),
        mo.hstack([
            mo.vstack([
                mo.md(translation('p3')),
                global_fee_growth
            ]),
            mo.vstack([
                mo.md(translation('p4')), 
                yard_takeover
            ])]),
        mo.hstack([
            mo.vstack([
                mo.md(translation('p5')), 
                average_bin_fullness
            ]),
            mo.vstack([
                mo.md(translation('p6')),
                scenario_selector
            ])
        ]),
        mo.hstack([
            mo.vstack([
                mo.md(translation('p7')),
                olo_multiplier
            ]),
            mo.vstack([
                mo.md(translation('p8')), 
                other_cost_baseline
            ])
        ]),
        mo.hstack([
            mo.vstack([
                mo.md(translation('p9')),
                organic_waste_pickup
            ]), 
            mo.vstack([
                mo.md(translation('p10')),
                olo_upload
            ])
        ])
    ])
    return (
        average_bin_fullness,
        global_fee_growth,
        olo_multiplier,
        olo_upload,
        organic_waste_pickup,
        other_cost_baseline,
        scenario_selector,
        yard_takeover,
    )


@app.cell(hide_code=True)
def _(mo, pd, translation):
    fee_hike = mo.ui.slider(0,100, show_value=True, include_input=True, value=30)
    remove_weekly = mo.ui.checkbox()
    forecast_periods = mo.ui.slider(0,20, show_value=True, include_input=True, value=5)

    ind_setup_df = pd.DataFrame(columns=[translation('t1'),translation('t2'),translation('t3')], data=[[30,5,False]])
    ind_setup_editor = mo.ui.data_editor(data=ind_setup_df)


    mo.vstack([
        mo.md(translation('s1')),
        mo.hstack([mo.md(translation('s2')), fee_hike], align='center'),
        mo.hstack([mo.md(translation('s3')), remove_weekly],  align='center'),
        mo.hstack([mo.md(translation('s4')), forecast_periods], align='center'),
        mo.md(translation('s5')),
        mo.md(translation('s6')),
        ind_setup_editor

    ])
    return fee_hike, forecast_periods, ind_setup_editor, remove_weekly


@app.cell(hide_code=True)
def _(mo, pd, translation):
    fee_hike_coop = mo.ui.slider(0,100, show_value=True, include_input=True, value=30)
    persons_per_bin_coop = mo.ui.slider(45,200,show_value=True, include_input=True, value=45)
    forecast_periods_coop = mo.ui.slider(0,20, show_value=True, include_input=True, value=5)

    coop_setup_df = pd.DataFrame(columns=[translation('t1'),translation('t4'),translation('t2')], data=[[30,45,5]])
    coop_setup_editor = mo.ui.data_editor(data=coop_setup_df)

    mo.vstack([
         mo.md(translation('s7')),
        mo.hstack([mo.md(translation('s8')), fee_hike_coop], align='center'),
        mo.hstack([mo.md(translation('s9')), persons_per_bin_coop], align='center'),
        mo.hstack([mo.md(translation('s10')), forecast_periods_coop], align='center'),
        mo.md(translation('s11')),
        coop_setup_editor
    ])
    return (
        coop_setup_editor,
        fee_hike_coop,
        forecast_periods_coop,
        persons_per_bin_coop,
    )


@app.cell(hide_code=True)
def _(
    mo,
    new_fee_1_simple,
    new_fee_1_step,
    new_fee_2_simple,
    new_fee_2_step,
    new_fee_3_simple,
    new_fee_3_step,
    new_fee_4_simple,
    new_fee_4_step,
    new_fee_5_simple,
    new_fee_5_step,
    new_fee_6_simple,
    new_fee_6_step,
    new_fee_7_simple,
    new_fee_7_step,
    old_fee_1,
    old_fee_2,
    old_fee_3,
    old_fee_4,
    old_fee_5,
    old_fee_6,
    old_fee_7,
    translation,
):
    mo.vstack([
        mo.md(translation('e1')),
        mo.md(translation('e2')),
        mo.hstack([
            mo.vstack([
                mo.md(translation('e3')),
                mo.stat(
                    label=translation('e4'),
                    value=f'{(old_fee_1/12.):.1f} € → {new_fee_1_simple/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_1_simple-old_fee_1)/12.:.1f} {translation("e5")}', 
                    bordered=True
                ), 
                mo.stat(
                    label=translation('e6'),
                    value=f'{(old_fee_2/12.):.1f} € → {new_fee_2_simple/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_2_simple-old_fee_2)/12.:.1f} {translation("e5")}', 
                    bordered=True
                )
                , 
                mo.stat(
                    label=translation('e7'),
                    value=f'{(old_fee_3/12.):.1f} € → {new_fee_3_simple/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_3_simple-old_fee_3)/12.:.1f} {translation("e5")}', 
                    bordered=True
                ), 
                mo.stat(
                    label=translation('e8'),
                    value=f'{(old_fee_4/12.):.1f} € → {new_fee_4_simple/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_4_simple-old_fee_4)/12.:.1f} {translation("e5")}', 
                    bordered=True
                )
            ]),
            mo.vstack([
                mo.md(translation('e9')),
                mo.stat(
                    label=translation('e4'),
                    value=f'{(old_fee_1/12.):.1f} € → {new_fee_1_step/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_1_step-old_fee_1)/12.:.1f} {translation("e5")}', 
                    bordered=True
                ), 
                mo.stat(
                    label=translation('e6'),
                    value=f'{(old_fee_2/12.):.1f} € → {new_fee_2_step/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_2_step-old_fee_2)/12.:.1f} {translation("e5")}', 
                    bordered=True
                )
                , 
                mo.stat(
                    label=translation('e7'),
                    value=f'{(old_fee_3/12.):.1f} € → {new_fee_3_step/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_3_step-old_fee_3)/12.:.1f} {translation("e5")}', 
                    bordered=True
                ), 
                mo.stat(
                    label=translation('e8'),
                    value=f'{(old_fee_4/12.):.1f} € → {new_fee_4_step/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_4_step-old_fee_4)/12.:.1f} {translation("e5")}', 
                    bordered=True
                )
            ])
        ]),
        mo.md(translation('e10')),
        mo.hstack([
            mo.vstack([
                mo.md(translation('e3')),
                mo.stat(
                    label=translation('e11'),
                    value=f'{(old_fee_5/12.):.1f} € → {new_fee_5_simple/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_5_simple-old_fee_5)/12.:.1f} {translation("e5")}', 
                    bordered=True
                ), 
                mo.stat(
                    label=translation('e12'),
                    value=f'{(old_fee_6/12.):.1f} € → {new_fee_6_simple/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_6_simple-old_fee_6)/12.:.1f} {translation("e5")}', 
                    bordered=True
                )
                , 
                mo.stat(
                    label=translation('e13'),
                    value=f'{(old_fee_7/12.):.1f} € → {new_fee_7_simple/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_7_simple-old_fee_7)/12.:.1f} {translation("e5")}', 
                    bordered=True
                )
            ]),
            mo.vstack([
                mo.md(translation('e9')),
                mo.stat(
                    label=translation('e11'),
                    value=f'{(old_fee_5/12.):.1f} € → {new_fee_5_step/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_5_step-old_fee_5)/12.:.1f} {translation("e5")}', 
                    bordered=True
                ), 
                mo.stat(
                    label=translation('e12'),
                    value=f'{(old_fee_6/12.):.1f} € → {new_fee_6_step/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_6_step-old_fee_6)/12.:.1f} {translation("e5")}', 
                    bordered=True
                )
                , 
                mo.stat(
                    label=translation('e13'),
                    value=f'{(old_fee_7/12.):.1f} € → {new_fee_7_step/12.:.1f} €', 
                    direction='increase',
                    caption=f'{(new_fee_7_step-old_fee_7)/12.:.1f} {translation("e5")}', 
                    bordered=True
                )
            ])
        ])
    ])
    return


@app.cell
def _(
    coop_setup_editor,
    fee_data,
    fee_hike,
    fee_hike_coop,
    ind_old_fees,
    ind_setup_editor,
    translation,
):
    # create model households
    ind_fee_hike_simple = (1 + fee_hike.value/100.)
    ind_fee_hike_step = (1 + ind_setup_editor.value[translation('t1')].div(100.)).product()
    ## 1 Individual household, 1x weekly 120L
    old_fee_1 = ind_old_fees[2][0]
    new_fee_1_simple = old_fee_1 * ind_fee_hike_simple
    new_fee_1_step = old_fee_1 * ind_fee_hike_step
    ## 2 Individual household, 2x month 120L
    old_fee_2 = ind_old_fees[1][0]
    new_fee_2_simple = old_fee_2 * ind_fee_hike_simple
    new_fee_2_step = old_fee_2 * ind_fee_hike_step

    _coop_sample = fee_data.query('Payer == "Coop"')
    _coop_baseline = _coop_sample.query(f'{translation("gr1")} == 2025').groupby(['IntervalPerWeek'], as_index=False)[['CollectionPoints','BinCount','TotalFee','TotalVolume']].sum()
    _coop_baseline['FeePerBin'] = _coop_baseline['TotalFee'].div(_coop_baseline['BinCount'])
    _coop_baseline['BinPerPoint'] = _coop_baseline['BinCount'].div(_coop_baseline['CollectionPoints'])
    _coop_baseline['FeePerPoint'] = _coop_baseline.FeePerBin.multiply(_coop_baseline.BinPerPoint)

    coop_fee_hike_simple = (1 + fee_hike_coop.value/100.)
    coop_fee_hike_step = (1 + coop_setup_editor.value[translation('t1')].div(100.)).product()
    ## 3 Block of flats, 2x weekly 1100L, 45 people per bin 
    old_fee_3 = _coop_baseline['FeePerBin'].values[2] / 45. * 2
    new_fee_3_simple = old_fee_3 * coop_fee_hike_simple
    new_fee_3_step = old_fee_3 * coop_fee_hike_step

    ## 4 Block of flats, 3x weekly 1100L, 60 people per bin
    old_fee_4 = _coop_baseline['FeePerBin'].values[3] / 60. * 2
    new_fee_4_simple = old_fee_4 * coop_fee_hike_simple
    new_fee_4_step = old_fee_4 * coop_fee_hike_step

    ## 5 120L bin, 2x monthly
    old_fee_5 = ind_old_fees[1][0]
    new_fee_5_simple = old_fee_5 * ind_fee_hike_simple
    new_fee_5_step = old_fee_5 * ind_fee_hike_step

    ## 6 240L bin, 2x monthly
    old_fee_6 = ind_old_fees[3][0]
    new_fee_6_simple = old_fee_6 * ind_fee_hike_simple
    new_fee_6_step = old_fee_6 * ind_fee_hike_step

    ## 7 1100L Bin, 2x weekly
    old_fee_7 = _coop_baseline['FeePerBin'].values[2]
    new_fee_7_simple = old_fee_7 * coop_fee_hike_simple
    new_fee_7_step = old_fee_7  * coop_fee_hike_step
    return (
        new_fee_1_simple,
        new_fee_1_step,
        new_fee_2_simple,
        new_fee_2_step,
        new_fee_3_simple,
        new_fee_3_step,
        new_fee_4_simple,
        new_fee_4_step,
        new_fee_5_simple,
        new_fee_5_step,
        new_fee_6_simple,
        new_fee_6_step,
        new_fee_7_simple,
        new_fee_7_step,
        old_fee_1,
        old_fee_2,
        old_fee_3,
        old_fee_4,
        old_fee_5,
        old_fee_6,
        old_fee_7,
    )


@app.cell(hide_code=True)
def _(alt, mo, organic_waste_pickup, results_overview, translation):
    results_together, detailed_results_together = results_overview()

    organic_schedule = organic_waste_pickup.selected_key

    def summary_chart(model, result_df):

        _df = result_df.query('Model == @model')[[translation("gr1"),translation('gr10'),translation("sc0")]].copy()
        _standard = translation('d5')
        _exp_df = result_df.query(f'Model == @model and {translation("sc0")} == @_standard').drop(columns=[translation('gr10'),translation('gr11'),translation("sc0")])
        _melt_exp = _exp_df.melt(id_vars=[translation("gr1")], value_vars=['1 OLO', translation('olo1'), translation('olo3'), translation('olo2')], var_name=translation('olo4'),value_name=translation('olo5')).sort_values(by=[translation('olo4')], ascending=True)

        lines_chart = alt.Chart(_df).mark_trail().encode(alt.X(translation("gr1")).axis(format='0d', 
                                                                             tickCount=_df[translation("gr1")].max()), 
                                                         y=translation('gr10'),
                                            color=alt.Color(translation("sc0")).scale(scheme='paired'))

        bar_chart = alt.Chart(_melt_exp).mark_bar(size=25).encode(alt.X(translation("gr1")).axis(format='0d', 
                                                                             tickCount=_df[translation("gr1")].max()),y=translation('olo5'),color=translation('olo4'))

        return lines_chart + bar_chart 


    total_simple = mo.ui.altair_chart(summary_chart(translation('sc2'), results_together), label=f"{translation('l1')} {organic_schedule})")
    total_stepped = mo.ui.altair_chart(summary_chart(translation('sc1'), results_together), label=f"{translation('l2')}  {organic_schedule})")

    export_button = mo.ui.run_button(label=translation('ex17'))

    mo.vstack([
        mo.md(translation('r1')),
        total_simple,
        total_stepped,
        export_button
    ])
    return export_button, results_together


@app.cell
def _(
    average_bin_fullness,
    coop_setup_editor,
    fee_hike,
    fee_hike_coop,
    global_fee_growth,
    ind_setup_editor,
    olo_multiplier,
    organic_waste_pickup,
    other_cost_baseline,
    pd,
    persons_per_bin_coop,
    remove_weekly,
    scenario_selector,
    translation,
    yard_takeover,
):
    def results_for_export(results_frame: pd.DataFrame) -> pd.DataFrame:

        cols = results_frame.columns
        buffer = [pd.NA] * len(cols)
        footer = [
            buffer,
            buffer, 
            [translation('ex1')] + [pd.NA] * (len(cols)-1),
            [translation('ex2')] + [pd.NA] * (len(cols)-1),
            [translation('ex3'), f'+{fee_hike.value} % | {translation('ex4') if remove_weekly.value else translation('ex5')}'] + [pd.NA] * (len(cols)-2),
            [translation('ex6'), f'+{fee_hike_coop.value} % | {persons_per_bin_coop.value} {translation('ex7')}']+ [pd.NA] * (len(cols)-2),
            buffer,
            [translation('ex8')] + [pd.NA] * (len(cols)-1),
            [translation('ex2')] + [pd.NA] * (len(cols)-1),
            [ind_setup_editor.value.to_string()] + [pd.NA] * (len(cols)-1),
            [translation('ex6')] + [pd.NA] * (len(cols)-1),
            [coop_setup_editor.value.to_string()] + [pd.NA] * (len(cols)-1),
            buffer,
            [translation('ex9')] + [pd.NA] * (len(cols)-1),
            [translation('ex10'), scenario_selector.selected_key] + [pd.NA] * (len(cols)-2),
            [translation('ex11'), organic_waste_pickup.selected_key] + [pd.NA] * (len(cols)-2),
            [translation('ex12'), yard_takeover.selected_key] + [pd.NA] * (len(cols)-2),
            [translation('ex13'), f'{global_fee_growth.value} %'] + [pd.NA] * (len(cols)-2),
            [translation('ex14'), f'{average_bin_fullness.value} %'] + [pd.NA] * (len(cols)-2),
            [translation('ex15'), f'{olo_multiplier.value} %'] + [pd.NA] * (len(cols)-2),
            [translation('ex16'), f'{other_cost_baseline.value:,.0f} EUR'] + [pd.NA] * (len(cols)-2)
        ]

        footer_df = pd.DataFrame(columns=cols, data=footer)

        return pd.concat([results_frame, footer_df], ignore_index=True)

    return (results_for_export,)


@app.cell
def _(export_button, mo, results_for_export, results_together, translation):
    mo.stop(not export_button.value)
    export_results = mo.ui.dataframe(results_for_export(results_together),
                                     show_download=True,
                                     download_csv_encoding='utf-8-sig')
    mo.vstack([
        mo.md(translation('r2')),
        mo.md(translation('r3')),
        mo.md(translation('r4')),
        mo.md(translation('r5')),
        export_results
    ])
    return


@app.cell
def _(OLO_base, organic_waste_pickup, translation):
    def update_organic(cost_df):
        organic_schedule = organic_waste_pickup.value
        if organic_schedule == 1: # 2x week even in winter 
            """
            Now we have 2x week from Apr to November = 35 weeks = 70 pickups 
            1x week from Dec to Mar = 17 weeks = 17 pickups 
            Total # of pickups = 87
            Going to full 2x increases the number of pickups by 17 to 104
            Cost increase is 104/87
            """
            cost_df['Organic waste collection'] = cost_df['Organic waste collection'] * (104./87.)
        elif organic_schedule == 2: # 1x week all year
            """
            Now we have 2x week from Apr to November = 35 weeks = 70 pickups 
            1x week from Dec to Mar = 17 weeks = 17 pickups 
            Total # of pickups = 87
            Going to 1x decreases the number of pickups by 35 to 52
            Cost fall is 52/87
            """
            cost_df['Organic waste collection'] = cost_df['Organic waste collection'] * (52./87.)
        cost_df['1 OLO'] = cost_df.drop(columns=['year']).sum(axis=1)
        cost_df[translation('gr1')] = cost_df.year - 2025
        return cost_df.query('year > 2024')



    OLO = update_organic(OLO_base.copy())
    return (OLO,)


@app.cell(hide_code=True)
def _(scenario_selector):
    SCENARIO = scenario_selector.value
    return (SCENARIO,)


@app.cell(hide_code=True)
def _(coop_results_, ind_df_results, pd, translation):
    results_summary = ind_df_results.query(f'{translation("gr1")} < 2').copy()
    results_summary[translation('gr3')] = translation('gr6')

    _coop_period1 = coop_results_.query(f'{translation("gr1")} < 2').copy()
    _coop_period1[translation('gr3')] = translation('gr7')

    results_summary = pd.concat([results_summary, _coop_period1], ignore_index=True)
    results_summary[translation("gr1")] = results_summary[translation("gr1")].astype(str).str.replace('0',translation("gr4")).str.replace('1',translation("gr5"))
    return (results_summary,)


@app.cell(hide_code=True)
def _(coop_results_, ind_df_results, mo, pd, results_summary, translation):
    last_period_ind = ind_df_results.groupby([translation("gr2")], as_index=False)[translation("gr1")].max()
    last_period_coop = coop_results_.groupby([translation("gr2")], as_index=False)[translation("gr1")].max()

    latest_ind = ind_df_results.merge(last_period_ind, on=[translation("gr2"),translation("gr1")], how='inner')
    latest_ind[translation("gr1")] = translation("gr8")
    latest_ind[translation("gr3")] = translation("gr6")
    latest_coop = coop_results_.merge(last_period_coop, on=[translation("gr2"),translation("gr1")], how='inner')
    latest_coop[translation("gr1")] = translation("gr8")
    latest_coop[translation("gr3")] = translation("gr7")

    results_summary_latest = pd.concat([results_summary, latest_ind, latest_coop], ignore_index=True)
    results_summary_latest[translation('gr10')] = results_summary_latest[translation('gr10')].round(0).astype(int)
    results_summary_totals = results_summary_latest.groupby([translation("gr2"),translation("gr1")], as_index=False)[translation('gr10')].sum()
    results_summary_totals[translation("gr3")] = translation("gr9")

    final_results = pd.concat([results_summary_latest, results_summary_totals], ignore_index=True).sort_values(by=[translation("gr2"), translation("gr3"), translation("gr1")])

    results_pivot = pd.pivot(final_results, columns=translation("gr1"), index=[translation("gr2"),translation("gr3")], values=[translation('gr10')])

    mo.vstack([
        mo.md(translation('r6')),
        mo.md(translation('r7')),
        mo.md(translation('r8')),
        mo.md(translation('r9')),
        results_pivot
    ])
    return


@app.cell(hide_code=True)
def _(alt, ind_df_results, mo, translation):
    ind_fee_chart = mo.ui.altair_chart(alt.Chart(ind_df_results).mark_trail().encode(x=translation('gr1'), y=translation('gr10'), color=translation('gr2')), label=translation('ir1'))

    ind_fee_chart
    return


@app.cell
def _(alt, coop_results_, mo, translation):
    coop_fee_chart = mo.ui.altair_chart(alt.Chart(coop_results_).mark_trail().encode(x=translation('gr1'), y=translation('gr10'), color=translation('gr2')), label=translation('cr3'))

    coop_bin_chart = mo.ui.altair_chart(alt.Chart(coop_results_).mark_trail().encode(x=translation('gr1'), y=translation('gr11'), color=translation('gr2')), label=translation('cr4'))

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
    translation,
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

        results = pd.DataFrame({translation('gr10'): fees,
                               translation('gr11'): simple_coop_forecast_bins.sum(axis=1),
                               translation('gr1'): periods,
                                translation('gr2'): translation('sc2')})
        return results 

    coop_results = coop_result_simple(coop_old_bin_ratio, coop_old_fees, coop_old_points, fee_hike_coop.value, forecast_periods_coop.value, people_per_bin=persons_per_bin_coop.value, scenario=SCENARIO)
    return coop_result_simple, coop_results


@app.cell
def _(SCENARIO, coop_results, indexer, np, pd, stepped_coop, translation):
    def coop_result_stepped(scenario: int = 1):

        stepped_coop_bins, stepped_coop_fees = stepped_coop(scenario)
        stepped_coop_bins_total = stepped_coop_bins.sum(axis=1)
        stepped_coop_fees_total = stepped_coop_fees.sum(axis=1)

        periods = np.arange(len(stepped_coop_fees_total))
        index_array = indexer(periods)
        stepped_coop_fees_total = stepped_coop_fees_total * index_array


        return pd.DataFrame(
            {translation('gr10'): stepped_coop_fees_total,
             translation('gr11'): stepped_coop_bins_total,
             translation('gr1'): periods,
             translation('gr2'): translation('sc1')})

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
def _(ind_old_fees, indexer, np, pd, run_individual, translation):
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
        {translation('gr10'): ind_fee_evolution_total,
         translation('gr11'): ind_bin_evolution_total,
         translation('gr1'): periods,
         translation('gr2'): translation('sc2')})

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
    translation,
):
    def stepped_individual(scenario:int = 1):

        old_fees = ind_old_fees.copy() 
        baseline_bins = ind_baseline_bins.copy() 
        bin_evolution = np.array([])
        fee_evolution = np.array([])
        price_hike = 0.

        for step in ind_setup_editor.value.iterrows():
            step_values = step[1]
            price_hike = price_hike + (step_values[translation('t1')] / 100.)
            _new_fees = old_fees * (1 + price_hike)
            _evolution = run_individual(baseline_bins, step_values[translation('t2')]+1, 0, price_increase = price_hike, remove_weekly = step_values[translation('t3')], scenario = scenario)
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
        {translation('gr10'): stepped_ind_fees_total,
         translation('gr11'): stepped_ind_bins_total,
         translation('gr1'): periods,
         translation('gr2'): translation('sc1')})

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
    translation,
    yard_takeover,
    yard_total,
):
    def equalize_frames(longer_frame: pd.DataFrame, shorter_frame: pd.DataFrame, col_name: str = 'Rok'):
        minmax = shorter_frame.Rok.max()
        delta = longer_frame.Rok.max() - minmax
        last_row = shorter_frame.query(f'{col_name} == @minmax').copy()
        for step in range(1, delta+1):
            last_row[col_name] = minmax + step 
            shorter_frame = pd.concat([shorter_frame, last_row])

        return shorter_frame

    def build_expenses():

        expenses = OLO.copy()

        expenses[translation('olo2')] = 0.

        if yard_takeover.value > 0:
            expenses = expenses.merge(yard_total[['year','yard_cost']], on='year', how='left')
            expenses[translation('olo2')] = expenses['yard_cost'] * olo_multiplier.value
            expenses.drop(columns=['yard_cost'], inplace=True)

        other_exp_base = other_cost_baseline.value 
        periods = expenses[translation("gr1")].values 
        index = indexer(periods)
        other_exp = index * other_exp_base

        expenses[translation('olo1')] = other_exp 
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
        COLS = [translation('gr10'),translation('gr11'),translation('gr1'),translation('gr2')]
        scenarios = {
            1: translation('d5'),
            2: translation('d6'),
            3: translation('d4')
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
            simple[translation('sc0')] = scenarios[scenario]
            simple_detail.append(simple)
            simple_group = simple.groupby([translation('sc0'),translation('gr2'),translation('gr1')], as_index=False)[[translation('gr10'),translation("gr11")]].sum()
            simple_results.append(simple_group)

            stepped = pd.concat([coop_stepped, ind_stepped], ignore_index=True)
            stepped[translation('sc0')] = scenarios[scenario]
            stepped_detail.append(stepped)
            stepped_group = stepped.groupby([translation('sc0'),translation('gr2'),translation('gr1')], as_index=False)[[translation('gr10'),translation("gr11")]].sum()
            stepped_results.append(stepped_group)

        # add NPC scenario 

        simple_indexer = indexer(np.arange(periods[0]))
        stepped_indexer = indexer(np.arange(periods[1]))
        start_fee = baseline_fees['TotalFee'].values[0]
        simple_npc_fees = start_fee * simple_indexer 
        stepped_npc_fees = start_fee * stepped_indexer 

        simple_npc_df = pd.DataFrame(data={
            translation('sc0'): [translation('sc3')]*periods[0],
            translation('gr2'): [translation('sc2')]*periods[0],
            translation('gr1'): np.arange(periods[0]),
            translation('gr10'): simple_npc_fees}, index=np.arange(periods[0]))

        stepped_npc_df = pd.DataFrame(data={
            translation('sc0'): [translation('sc3')]*periods[1],
            translation('gr2'): [translation('sc1')]*periods[1],
            translation('gr1'): np.arange(periods[1]),
            translation('gr10'): stepped_npc_fees}, index=np.arange(periods[1]))


        together = pd.concat(simple_results+stepped_results+[simple_npc_df, stepped_npc_df], ignore_index=True)
        together_detail = pd.concat(simple_detail+stepped_detail, ignore_index=True)

        # merge OLO
        expenses = build_expenses()
        together = together.merge(expenses, on=[translation('gr1')], how='left')
        together[translation('olo3')] = together[translation("gr10")] * 0.1

        return together, together_detail

    return (results_overview,)


@app.cell
def _(
    coop_old_bin_ratio,
    coop_old_fees,
    coop_old_points,
    coop_setup_editor,
    np,
    translation,
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
            price_hike = price_hike + (step_values[translation('t1')] / 100.)
            _new_fees = old_fees * (1 + price_hike)
            _evolution, _latest_ratio = bin_per_point_forecast(baseline_ratio, baseline_points, price_hike, step_values[translation('t2')]+1, step_values[translation('t4')], scenario=scenario)
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
