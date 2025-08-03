import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io


def plot_scaling_results():
    """
    Plot scaling experiment results showing performance and timing vs system size.
    Uses CSV data with different sparsity levels for StructureFlow and NGM-NODE methods.
    """

    # CSV data from the experiment results (multiple seeds)
    csv_data = """num_vars,seed,sparsity_target,sparsity_actual,true_edges,edge_density,max_eigenvalue_real,Correlation-pearson_AUROC,Correlation-pearson_AUPRC,Correlation-pearson_training_time,Correlation-pearson_num_true_edges,StructureFlow_AUROC,StructureFlow_AUPRC,StructureFlow_training_time,StructureFlow_num_true_edges,NGM-NODE_AUROC,NGM-NODE_AUPRC,NGM-NODE_training_time,NGM-NODE_num_true_edges,ReferenceFitting_AUROC,ReferenceFitting_AUPRC,ReferenceFitting_training_time,ReferenceFitting_num_true_edges,experiment_id,total_experiments
10,4852,0.05,0.06666666666666667,6,0.06666666666666667,0.0,0.9734042553191489,0.7028138528138528,0.00824594497680664,6,0.9627659574468085,0.509090909090909,51.46,6,0.7810283687943262,0.3880834899749373,322.7519874572754,6.0,0.9592198581560284,0.45092592592592595,3.6239442825317383,6,1,54
10,4852,0.2,0.26666666666666666,24,0.26666666666666666,0.395655448780167,0.8135964912280701,0.5441361603475161,0.00867915153503418,24,0.7757675438596491,0.4692447182908081,57.832767486572266,24,0.7785087719298245,0.47540794937623915,324.6383876800537,24.0,0.8240131578947368,0.514064295997455,3.4556427001953125,24,2,54
10,4852,0.4,0.4111111111111111,37,0.4111111111111111,1.318209686662155,0.8631488631488632,0.7907212834471284,0.00930023193359375,37,0.8421278421278421,0.7562902634055009,57.44078350067139,37,0.752895752895753,0.6563467965949508,328.1729917526245,37.0,0.8704418704418705,0.7913406247469861,3.46559476852417,37,3,54
25,4852,0.05,0.043333333333333335,26,0.043333333333333335,0.5922078331269705,0.9792603056376012,0.5439231326379298,0.16530323028564453,26,0.9780724284063181,0.5028348937550893,69.76,26,0.7698407602414281,0.17294465805593287,1063.5547759532928,26.0,0.9750224733530243,0.49334740456368503,3.4929757118225098,26,4,54
25,4852,0.2,0.20333333333333334,122,0.20333333333333334,1.4305823939895395,0.8249193364403741,0.5595433355443513,0.06273722648620605,122,0.834493041749503,0.5292155372923445,77.5687415599823,122,0.7024655346608872,0.37147276335234236,997.8577566146851,122.0,0.7667112081608708,0.44678819797735214,4.441559791564941,122,5,54
25,4852,0.4,0.425,255,0.425,1.9448587728752713,0.7307154213036565,0.6671333191730175,0.06077694892883301,255,0.716120826709062,0.6356068852578105,77.48430800437927,255,0.5623635400105989,0.4655221440980293,824.9652671813965,255.0,0.6054478007419184,0.4892427504811191,3.8342349529266357,255,6,54
50,4852,0.05,0.05061224489795919,124,0.05061224489795919,0.902900847379586,0.9500753502769631,0.5227702410383083,0.3219568729400635,124,0.9247023324644292,0.47501918118317865,113.89,124,0.5977449223416965,0.0778863190985196,3123.374433517456,124.0,0.827491989790377,0.2605758567268615,4.696411848068237,124,7,54
50,4852,0.2,0.20530612244897958,503,0.20530612244897958,2.3717790475785745,0.8656085519929995,0.6316451281948303,0.4170088768005371,503,0.812686226158323,0.5528893190794936,119.93507289886475,503,0.5411253062496328,0.23896093426619205,2550.32168841362,503.0,0.6092687739362523,0.2935042195374286,7.3686792850494385,503,8,54
50,4852,0.4,0.4008163265306122,982,0.4008163265306122,2.951828649942834,0.7080847883778904,0.6359765607805926,0.5743980407714844,982,0.642092916233977,0.5640048208068421,119.54320907592773,982,0.5557294140376581,0.4414136447103131,1409.912146806717,982.0,0.5635000496419074,0.461288725514825,4.659662961959839,982,9,54
100,4852,0.05,0.05141414141414141,509,0.05141414141414141,1.5918627066326787,0.9705788898551185,0.5697944849005392,1.7494568824768066,509,0.9605501561918136,0.5352397389531048,184.26,509,0.5396871692528896,0.06193179938167517,10360.619023323059,509.0,0.594919931383656,0.12461957862532796,6.476845026016235,509,10,54
100,4852,0.2,0.2006060606060606,1986,0.2006060606060606,3.2186210953942522,0.8243974354044572,0.5559764380525823,1.7908899784088135,1986,0.7595058659933234,0.4685942391145377,231.6057665348053,1986,0.5109463210278287,0.2035805347143642,3091.8110506534576,1986.0,0.515307991980801,0.21396723750945168,5.281699895858765,1986,11,54
100,4852,0.4,0.39616161616161616,3922,0.39616161616161616,4.323132007151697,0.6578295686586025,0.5801396233834728,1.7575359344482422,3922,0.6195861248944748,0.5296250846965184,233.59105587005615,3922,0.5092083552941457,0.3973237266569635,2388.5191402435303,3922.0,0.4965277585507056,0.3882980426927889,5.058599948883057,3922,12,54
200,4852,0.05,0.05050251256281407,2010,0.05050251256281407,2.294999857440941,0.96814073879091,0.5526191264144195,7.23038387298584,2010,0.9489784297779332,0.5005987530279135,457.13,2010,,,,,0.4881887089951663,0.054118841674638346,5.88510274887085,2010,13,54
200,4852,0.2,0.2014070351758794,8016,0.2014070351758794,5.244219711505874,0.7674918500293061,0.4897851792773821,6.939878940582275,8016,0.6868925843441931,0.38349873985718347,816.0332379341125,8016,,,,,0.5153939167063573,0.21812220123540865,5.722508907318115,8016,14,54
200,4852,0.4,0.4037437185929648,16069,0.4037437185929648,6.501912281236287,0.6218929139158376,0.5434334918619736,7.355964183807373,16069,0.5659527723198657,0.4721297261061206,733.2379455566406,16069,,,,,0.5266332181363029,0.42794069682047264,5.874831914901733,16069,15,54
500,4852,0.05,0.04988777555110221,12447,0.04988777555110221,3.8611548089595615,0.9566752835561358,0.5056815473923015,61.566709995269775,12447,0.8569096186142633,0.3450751614951112,2454.66,12447,,,,,0.5679223335939249,0.09860205467582925,15.228864192962646,12447,16,54
500,4852,0.2,0.19991583166332666,49879,0.19991583166332666,7.857046880043978,0.690413197464293,0.3972364107787436,58.26046013832092,49879,0.5289162359044275,0.22013346016343596,5084.449959278107,49879,,,,,0.5775005197618788,0.2814493758006428,16.369847774505615,49879,17,54
500,4852,0.4,0.4006733466933868,99968,0.4006733466933868,10.29850094782119,0.5797805580917403,0.49229326803046286,60.129616022109985,99968,0.5104833479061066,0.4090570671710845,5428.748547077179,99968,,,,,0.5391311759294999,0.4398292752075346,15.862734079360962,99968,18,54
10,2502,0.05,0.03333333333333333,3,0.03333333333333333,0.0,0.9896907216494845,0.7555555555555555,0.014358043670654297,3,0.986254295532646,0.7,51.46,3,0.9536,0.7143,316.0287,3.0,0.986254295532646,0.7,3.9116318225860596,3,37,54
10,2502,0.2,0.14444444444444443,13,0.14444444444444443,0.45274256050913675,0.9336870026525199,0.6670541199116324,0.011578798294067383,13,0.9332449160035367,0.6064580347189044,57.0928475856781,13,0.7780725022104332,0.43960817162269067,328.5199406147003,13.0,0.9257294429708224,0.5489945125480242,3.117793083190918,13,20,54
10,2502,0.4,0.3333333333333333,30,0.3333333333333333,0.8311741245831213,0.8638095238095238,0.7129846759193442,0.011166095733642578,30,0.8390476190476192,0.6508294299766001,56.757171869277954,30,0.7552380952380953,0.5807710817767366,336.25325417518616,30.0,0.8247619047619047,0.655297259959569,2.91511607170105,30,21,54
25,2502,0.05,0.05333333333333334,32,0.05333333333333334,0.7196140462937771,0.9381323777403034,0.5174637896473324,0.05647420883178711,32,0.9517811973018548,0.5160586468732333,69.76,32,0.692295531197302,0.23074941235248975,1068.1875479221344,32.0,0.9520446880269814,0.52883216533079,3.7061920166015625,32,22,54
25,2502,0.2,0.19833333333333333,119,0.19833333333333333,1.312350516959753,0.8755272860132195,0.5792912693317145,0.06208515167236328,119,0.8422792041717873,0.5339294313329173,78.74769520759583,119,0.6186601122662504,0.3039079820751982,1081.0779702663422,119.0,0.7595077556714385,0.4585875761686677,3.6421401500701904,119,23,54
25,2502,0.4,0.385,231,0.385,1.7329931890341117,0.7258663502318325,0.6388499507982699,0.059736013412475586,231,0.7234766079943745,0.6233801558082154,79.93856620788574,231,0.6033632188454524,0.4766637796190075,995.7301144599915,231.0,0.6515920627595755,0.5152831437819262,3.7372992038726807,231,24,54
50,2502,0.05,0.05061224489795919,124,0.05061224489795919,1.0110962062906572,0.9766515966112741,0.570391982943973,0.3245840072631836,124,0.9628102259150646,0.5027629030484266,113.89,124,0.6237271912675139,0.09157567784636376,3192.3506700992584,124.0,0.818762218963832,0.3506930642437446,3.890043020248413,124,25,54
50,2502,0.2,0.20489795918367346,502,0.20489795918367346,2.2483800453103724,0.8555806802818755,0.5841160314833563,0.3926069736480713,502,0.8002683958859258,0.5185753101194658,121.33024144172668,502,0.5446083533732936,0.2342731467410447,2368.58771276474,502.0,0.6128409285779804,0.293764141167148,4.032464981079102,502,26,54
50,2502,0.4,0.4048979591836735,992,0.4048979591836735,2.864936135818006,0.704673863052965,0.6335978337504595,0.34891486167907715,992,0.6435709816462736,0.5677720499751625,119.8461434841156,992,0.5446442895952769,0.4405993383653992,1217.2671146392822,992.0,0.5678538386668949,0.45575983249566326,3.6820859909057617,992,27,54
100,2502,0.05,0.05171717171717172,512,0.05171717171717172,1.6213753680828042,0.9735850548060708,0.5677566445259788,1.7340631484985352,512,0.9701820103354237,0.528340874465576,184.26,512,0.5172116052843064,0.05503360508468549,10017.93087887764,512.0,0.5747142360745152,0.08098596863835426,4.164529085159302,512,28,54
100,2502,0.2,0.20656565656565656,2045,0.20656565656565656,3.2842529847500033,0.8198650415924539,0.5568001360244068,1.69008207321167,2045,0.744290976596657,0.4645960126865602,232.7276487350464,2045,0.5122506027947547,0.2101899865277203,3071.490483045578,2045.0,0.5212285487284065,0.21756823609129838,4.287989854812622,2045,29,54
100,2502,0.4,0.40545454545454546,4014,0.40545454545454546,4.275314334045113,0.6645226505093849,0.5940594166285376,1.7266550064086914,4014,0.6194301609918242,0.5382326014982242,234.22895073890686,4014,0.5131996457104444,0.41053888331431937,2378.5854711532593,4014.0,0.5162929163231064,0.4132647038026137,4.363715171813965,4014,30,54
200,2502,0.05,0.051457286432160805,2048,0.051457286432160805,2.4913927807530096,0.9694603056368901,0.549223169181854,7.430928945541382,2048,0.9524225167440122,0.4961120628161967,457.13,2048,,,,,0.5051248560064358,0.05870375206120263,5.979673147201538,2048,31,54
200,2502,0.2,0.19924623115577889,7930,0.19924623115577889,4.66541396943789,0.7738436412151697,0.490576798241162,6.874438762664795,7930,0.6866895221715108,0.379842272121609,571.0319414138794,7930,,,,,0.5142887948184072,0.2162142300729223,5.52838397026062,7930,32,54
200,2502,0.4,0.39977386934673365,15911,0.39977386934673365,6.696018599710984,0.6306827180548562,0.5518513860566123,6.833057165145874,15911,0.574681301138012,0.47429580458434983,571.5179307460785,15911,,,,,0.5250170802641689,0.42397067753035056,6.033360719680786,15911,33,54
500,2502,0.05,0.050569138276553106,12617,0.050569138276553106,3.9178266871702925,0.954972984577916,0.5059281437135656,57.61861705780029,12617,0.5649417194396844,0.09744909257470831,2454.66,12617,,,,,0.5649417194396844,0.09744909257470831,15.736231327056885,12617,34,54
500,2502,0.2,0.2014188376753507,50254,0.2014188376753507,8.18031517030121,0.6895586017834803,0.3970475162183286,60.14372205734253,50254,0.5291851691465888,0.2208703164645075,5353.444791316986,50254,,,,,0.5737566990254326,0.27832127167193405,17.218700647354126,50254,35,54
500,2502,0.4,0.40139078156312624,100147,0.40139078156312624,10.29850094782119,0.5797805580917403,0.49246649707812357,59.82843017578125,99754,0.5439168069596729,0.4431596091675929,5315.9313,99754,,,,,0.5439168069596729,0.4431596091675929,17.69760298728943,99754,36,54"""

    df = pd.read_csv(io.StringIO(csv_data))

    # Calculate means and standard errors across seeds for each system size and sparsity
    def calculate_stats(group):
        stats = {}
        for col in [
            "StructureFlow_AUROC",
            "StructureFlow_AUPRC",
            "StructureFlow_training_time",
            "NGM-NODE_AUROC",
            "NGM-NODE_AUPRC",
            "NGM-NODE_training_time",
            "ReferenceFitting_AUROC",
            "ReferenceFitting_AUPRC",
            "ReferenceFitting_training_time",
        ]:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) > 0:
                    stats[f"{col}_mean"] = values.mean()
                    stats[f"{col}_std"] = values.std()
                    stats[f"{col}_count"] = len(values)
                else:
                    stats[f"{col}_mean"] = np.nan
                    stats[f"{col}_std"] = np.nan
                    stats[f"{col}_count"] = 0
        return pd.Series(stats)

    # Group by system size and sparsity target, then calculate statistics
    stats_df = (
        df.groupby(["num_vars", "sparsity_target"]).apply(calculate_stats).reset_index()
    )
    stats_df = stats_df.rename(
        columns={"num_vars": "system_size", "sparsity_target": "sparsity"}
    )

    return stats_df


def plot_scaling_results_with_confidence():
    """
    Create plots with confidence intervals using alpha fills.
    """
    stats_df = plot_scaling_results()

    # Set up the plotting style
    plt.style.use("default")

    # Define red color palette for sparsities and line styles for models
    sparsity_colors = {
        0.05: "#8B0000",
        0.2: "#DC143C",
        0.4: "#FF6347",
    }  # Dark red to light red
    method_styles = {"StructureFlow": "-", "NGM-NODE": "--", "ReferenceFitting": ":"}
    sparsity_labels = {0.05: "5% sparse", 0.2: "20% sparse", 0.4: "40% sparse"}

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: AUROC vs System Size with confidence intervals
    for sparsity in [0.05, 0.2, 0.4]:
        sparsity_data = stats_df[stats_df["sparsity"] == sparsity]
        color = sparsity_colors[sparsity]

        # StructureFlow
        x_sf = sparsity_data["system_size"]
        y_sf = sparsity_data["StructureFlow_AUROC_mean"]
        y_err_sf = sparsity_data["StructureFlow_AUROC_std"] / np.sqrt(
            sparsity_data["StructureFlow_AUROC_count"]
        )
        ax1.plot(
            x_sf,
            y_sf,
            color=color,
            linestyle=method_styles["StructureFlow"],
            linewidth=2,
            marker="o",
            markersize=6,
            label=f"StructureFlow ({sparsity_labels[sparsity]})",
        )
        ax1.fill_between(x_sf, y_sf - y_err_sf, y_sf + y_err_sf, color=color, alpha=0.2)

        # NGM-NODE (filter out NaN values)
        ngm_data = sparsity_data.dropna(subset=["NGM-NODE_AUROC_mean"])
        x_ngm = ngm_data["system_size"]
        y_ngm = ngm_data["NGM-NODE_AUROC_mean"]
        y_err_ngm = ngm_data["NGM-NODE_AUROC_std"] / np.sqrt(
            ngm_data["NGM-NODE_AUROC_count"]
        )
        ax1.plot(
            x_ngm,
            y_ngm,
            color=color,
            linestyle=method_styles["NGM-NODE"],
            linewidth=2,
            marker="s",
            markersize=6,
            label=f"NGM-NODE ({sparsity_labels[sparsity]})",
        )
        ax1.fill_between(
            x_ngm, y_ngm - y_err_ngm, y_ngm + y_err_ngm, color=color, alpha=0.2
        )

        # Reference Fitting (filter out NaN values)
        rf_data = sparsity_data.dropna(subset=["ReferenceFitting_AUROC_mean"])
        x_rf = rf_data["system_size"]
        y_rf = rf_data["ReferenceFitting_AUROC_mean"]
        y_err_rf = rf_data["ReferenceFitting_AUROC_std"] / np.sqrt(
            rf_data["ReferenceFitting_AUROC_count"]
        )
        ax1.plot(
            x_rf,
            y_rf,
            color=color,
            linestyle=method_styles["ReferenceFitting"],
            linewidth=2,
            marker="^",
            markersize=6,
            label=f"Reference Fitting ({sparsity_labels[sparsity]})",
        )
        ax1.fill_between(x_rf, y_rf - y_err_rf, y_rf + y_err_rf, color=color, alpha=0.2)

    ax1.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title("AUROC vs System Size", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_xscale("log")
    ax1.set_xticks([10, 25, 50, 100, 200, 500])
    ax1.set_xticklabels([10, 25, 50, 100, 200, 500])

    # Plot 2: AUPRC vs System Size with confidence intervals
    for sparsity in [0.05, 0.2, 0.4]:
        sparsity_data = stats_df[stats_df["sparsity"] == sparsity]
        color = sparsity_colors[sparsity]

        # StructureFlow
        x_sf = sparsity_data["system_size"]
        y_sf = sparsity_data["StructureFlow_AUPRC_mean"]
        y_err_sf = sparsity_data["StructureFlow_AUPRC_std"] / np.sqrt(
            sparsity_data["StructureFlow_AUPRC_count"]
        )
        ax2.plot(
            x_sf,
            y_sf,
            color=color,
            linestyle=method_styles["StructureFlow"],
            linewidth=2,
            marker="o",
            markersize=6,
            label=f"StructureFlow ({sparsity_labels[sparsity]})",
        )
        ax2.fill_between(x_sf, y_sf - y_err_sf, y_sf + y_err_sf, color=color, alpha=0.2)

        # NGM-NODE (filter out NaN values)
        ngm_data = sparsity_data.dropna(subset=["NGM-NODE_AUPRC_mean"])
        x_ngm = ngm_data["system_size"]
        y_ngm = ngm_data["NGM-NODE_AUPRC_mean"]
        y_err_ngm = ngm_data["NGM-NODE_AUPRC_std"] / np.sqrt(
            ngm_data["NGM-NODE_AUPRC_count"]
        )
        ax2.plot(
            x_ngm,
            y_ngm,
            color=color,
            linestyle=method_styles["NGM-NODE"],
            linewidth=2,
            marker="s",
            markersize=6,
            label=f"NGM-NODE ({sparsity_labels[sparsity]})",
        )
        ax2.fill_between(
            x_ngm, y_ngm - y_err_ngm, y_ngm + y_err_ngm, color=color, alpha=0.2
        )

        # Reference Fitting (filter out NaN values)
        rf_data = sparsity_data.dropna(subset=["ReferenceFitting_AUPRC_mean"])
        x_rf = rf_data["system_size"]
        y_rf = rf_data["ReferenceFitting_AUPRC_mean"]
        y_err_rf = rf_data["ReferenceFitting_AUPRC_std"] / np.sqrt(
            rf_data["ReferenceFitting_AUPRC_count"]
        )
        ax2.plot(
            x_rf,
            y_rf,
            color=color,
            linestyle=method_styles["ReferenceFitting"],
            linewidth=2,
            marker="^",
            markersize=6,
            label=f"Reference Fitting ({sparsity_labels[sparsity]})",
        )
        ax2.fill_between(x_rf, y_rf - y_err_rf, y_rf + y_err_rf, color=color, alpha=0.2)

    ax2.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax2.set_ylabel("AUPRC", fontsize=12)
    ax2.set_title("AUPRC vs System Size", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xscale("log")
    ax2.set_xticks([10, 25, 50, 100, 200, 500])
    ax2.set_xticklabels([10, 25, 50, 100, 200, 500])

    # Plot 3: Training Time vs System Size (averaged over sparsity levels)
    sf_avg_data = (
        stats_df.groupby("system_size")["StructureFlow_training_time_mean"]
        .mean()
        .reset_index()
    )
    ax3.plot(
        sf_avg_data["system_size"],
        sf_avg_data["StructureFlow_training_time_mean"],
        color="#8B0000",
        linestyle="-",
        marker="o",
        linewidth=2,
        markersize=8,
        label="StructureFlow",
    )

    ngm_avg_data = (
        stats_df.dropna(subset=["NGM-NODE_training_time_mean"])
        .groupby("system_size")["NGM-NODE_training_time_mean"]
        .mean()
        .reset_index()
    )
    ax3.plot(
        ngm_avg_data["system_size"],
        ngm_avg_data["NGM-NODE_training_time_mean"],
        color="#8B0000",
        linestyle="--",
        marker="s",
        linewidth=2,
        markersize=8,
        label="NGM-NODE",
    )

    rf_avg_data = (
        stats_df.dropna(subset=["ReferenceFitting_training_time_mean"])
        .groupby("system_size")["ReferenceFitting_training_time_mean"]
        .mean()
        .reset_index()
    )
    ax3.plot(
        rf_avg_data["system_size"],
        rf_avg_data["ReferenceFitting_training_time_mean"],
        color="#8B0000",
        linestyle=":",
        marker="^",
        linewidth=2,
        markersize=8,
        label="Reference Fitting",
    )

    ax3.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax3.set_ylabel("Training Time (seconds)", fontsize=12)
    ax3.set_title("Training Time vs System Size", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=9, loc="best")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")
    ax3.set_xscale("log")
    ax3.set_xticks([10, 25, 50, 100, 200, 500])
    ax3.set_xticklabels([10, 25, 50, 100, 200, 500])

    plt.tight_layout()
    plt.savefig(
        "scaling_experiment_plots_with_confidence.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig("scaling_experiment_plots_with_confidence.pdf", bbox_inches="tight")
    print(
        "Plots with confidence intervals saved to scaling_experiment_plots_with_confidence.png and .pdf"
    )
    plt.show()

    return fig, stats_df


def plot_scaling_results_clean():
    """
    Create clean plots without confidence intervals.
    """
    stats_df = plot_scaling_results()

    # Set up the plotting style
    plt.style.use("default")

    # Define red color palette for sparsities and line styles for models
    sparsity_colors = {
        0.05: "#8B0000",
        0.2: "#DC143C",
        0.4: "#FF6347",
    }  # Dark red to light red
    method_styles = {"StructureFlow": "-", "NGM-NODE": "--", "ReferenceFitting": ":"}
    sparsity_labels = {0.05: "5% sparse", 0.2: "20% sparse", 0.4: "40% sparse"}

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: AUROC vs System Size
    for sparsity in [0.05, 0.2, 0.4]:
        sparsity_data = stats_df[stats_df["sparsity"] == sparsity]
        color = sparsity_colors[sparsity]

        # StructureFlow
        x_sf = sparsity_data["system_size"]
        y_sf = sparsity_data["StructureFlow_AUROC_mean"]
        ax1.plot(
            x_sf,
            y_sf,
            color=color,
            linestyle=method_styles["StructureFlow"],
            linewidth=2,
            marker="o",
            markersize=6,
            label=f"StructureFlow ({sparsity_labels[sparsity]})",
        )

        # NGM-NODE (filter out NaN values)
        ngm_data = sparsity_data.dropna(subset=["NGM-NODE_AUROC_mean"])
        x_ngm = ngm_data["system_size"]
        y_ngm = ngm_data["NGM-NODE_AUROC_mean"]
        ax1.plot(
            x_ngm,
            y_ngm,
            color=color,
            linestyle=method_styles["NGM-NODE"],
            linewidth=2,
            marker="s",
            markersize=6,
            label=f"NGM-NODE ({sparsity_labels[sparsity]})",
        )

        # Reference Fitting (filter out NaN values)
        rf_data = sparsity_data.dropna(subset=["ReferenceFitting_AUROC_mean"])
        x_rf = rf_data["system_size"]
        y_rf = rf_data["ReferenceFitting_AUROC_mean"]
        ax1.plot(
            x_rf,
            y_rf,
            color=color,
            linestyle=method_styles["ReferenceFitting"],
            linewidth=2,
            marker="^",
            markersize=6,
            label=f"Reference Fitting ({sparsity_labels[sparsity]})",
        )

    ax1.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title("AUROC vs System Size", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_xscale("log")
    ax1.set_xticks([10, 25, 50, 100, 200, 500])
    ax1.set_xticklabels([10, 25, 50, 100, 200, 500])

    # Plot 2: AUPRC vs System Size
    for sparsity in [0.05, 0.2, 0.4]:
        sparsity_data = stats_df[stats_df["sparsity"] == sparsity]
        color = sparsity_colors[sparsity]

        # StructureFlow
        x_sf = sparsity_data["system_size"]
        y_sf = sparsity_data["StructureFlow_AUPRC_mean"]
        ax2.plot(
            x_sf,
            y_sf,
            color=color,
            linestyle=method_styles["StructureFlow"],
            linewidth=2,
            marker="o",
            markersize=6,
            label=f"StructureFlow ({sparsity_labels[sparsity]})",
        )

        # NGM-NODE (filter out NaN values)
        ngm_data = sparsity_data.dropna(subset=["NGM-NODE_AUPRC_mean"])
        x_ngm = ngm_data["system_size"]
        y_ngm = ngm_data["NGM-NODE_AUPRC_mean"]
        ax2.plot(
            x_ngm,
            y_ngm,
            color=color,
            linestyle=method_styles["NGM-NODE"],
            linewidth=2,
            marker="s",
            markersize=6,
            label=f"NGM-NODE ({sparsity_labels[sparsity]})",
        )

        # Reference Fitting (filter out NaN values)
        rf_data = sparsity_data.dropna(subset=["ReferenceFitting_AUPRC_mean"])
        x_rf = rf_data["system_size"]
        y_rf = rf_data["ReferenceFitting_AUPRC_mean"]
        ax2.plot(
            x_rf,
            y_rf,
            color=color,
            linestyle=method_styles["ReferenceFitting"],
            linewidth=2,
            marker="^",
            markersize=6,
            label=f"Reference Fitting ({sparsity_labels[sparsity]})",
        )

    ax2.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax2.set_ylabel("AUPRC", fontsize=12)
    ax2.set_title("AUPRC vs System Size", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xscale("log")
    ax2.set_xticks([10, 25, 50, 100, 200, 500])
    ax2.set_xticklabels([10, 25, 50, 100, 200, 500])

    # Plot 3: Training Time vs System Size (averaged over sparsity levels)
    sf_avg_data = (
        stats_df.groupby("system_size")["StructureFlow_training_time_mean"]
        .mean()
        .reset_index()
    )
    ax3.plot(
        sf_avg_data["system_size"],
        sf_avg_data["StructureFlow_training_time_mean"],
        color="#8B0000",
        linestyle="-",
        marker="o",
        linewidth=2,
        markersize=8,
        label="StructureFlow",
    )

    ngm_avg_data = (
        stats_df.dropna(subset=["NGM-NODE_training_time_mean"])
        .groupby("system_size")["NGM-NODE_training_time_mean"]
        .mean()
        .reset_index()
    )
    ax3.plot(
        ngm_avg_data["system_size"],
        ngm_avg_data["NGM-NODE_training_time_mean"],
        color="#8B0000",
        linestyle="--",
        marker="s",
        linewidth=2,
        markersize=8,
        label="NGM-NODE",
    )

    rf_avg_data = (
        stats_df.dropna(subset=["ReferenceFitting_training_time_mean"])
        .groupby("system_size")["ReferenceFitting_training_time_mean"]
        .mean()
        .reset_index()
    )
    ax3.plot(
        rf_avg_data["system_size"],
        rf_avg_data["ReferenceFitting_training_time_mean"],
        color="#8B0000",
        linestyle=":",
        marker="^",
        linewidth=2,
        markersize=8,
        label="Reference Fitting",
    )

    ax3.set_xlabel("System Size (Number of Variables)", fontsize=12)
    ax3.set_ylabel("Training Time (seconds)", fontsize=12)
    ax3.set_title("Training Time vs System Size", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=9, loc="best")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")
    ax3.set_xscale("log")
    ax3.set_xticks([10, 25, 50, 100, 200, 500])
    ax3.set_xticklabels([10, 25, 50, 100, 200, 500])

    plt.tight_layout()
    plt.savefig("scaling_experiment_plots_clean.png", dpi=300, bbox_inches="tight")
    plt.savefig("scaling_experiment_plots_clean.pdf", bbox_inches="tight")
    print("Clean plots saved to scaling_experiment_plots_clean.png and .pdf")
    plt.show()

    return fig, stats_df


def plot_scaling_results_summary():
    """
    Print summary statistics for the scaling experiment.
    """
    stats_df = plot_scaling_results()

    sparsity_labels = {0.05: "5% sparse", 0.2: "20% sparse", 0.4: "40% sparse"}

    print("\n" + "=" * 80)
    print("SCALING EXPERIMENT SUMMARY")
    print("=" * 80)

    for sparsity in [0.05, 0.2, 0.4]:
        print(f"\nSparsity: {sparsity_labels[sparsity]}")
        print("-" * 40)
        sparsity_data = stats_df[stats_df["sparsity"] == sparsity]

        for _, row in sparsity_data.iterrows():
            system_size = int(row["system_size"])
            sf_auroc = row["StructureFlow_AUROC_mean"]
            sf_auroc_std = row["StructureFlow_AUROC_std"]
            sf_auprc = row["StructureFlow_AUPRC_mean"]
            sf_auprc_std = row["StructureFlow_AUPRC_std"]
            sf_time = row["StructureFlow_training_time_mean"]
            sf_time_std = row["StructureFlow_training_time_std"]
            ngm_auroc = (
                row["NGM-NODE_AUROC_mean"]
                if pd.notna(row["NGM-NODE_AUROC_mean"])
                else None
            )
            ngm_auroc_std = (
                row["NGM-NODE_AUROC_std"]
                if pd.notna(row["NGM-NODE_AUROC_std"])
                else None
            )
            ngm_auprc = (
                row["NGM-NODE_AUPRC_mean"]
                if pd.notna(row["NGM-NODE_AUPRC_mean"])
                else None
            )
            ngm_auprc_std = (
                row["NGM-NODE_AUPRC_std"]
                if pd.notna(row["NGM-NODE_AUPRC_std"])
                else None
            )
            ngm_time = (
                row["NGM-NODE_training_time_mean"]
                if pd.notna(row["NGM-NODE_training_time_mean"])
                else None
            )
            ngm_time_std = (
                row["NGM-NODE_training_time_std"]
                if pd.notna(row["NGM-NODE_training_time_std"])
                else None
            )

            rf_auroc = (
                row["ReferenceFitting_AUROC_mean"]
                if pd.notna(row["ReferenceFitting_AUROC_mean"])
                else None
            )
            rf_auroc_std = (
                row["ReferenceFitting_AUROC_std"]
                if pd.notna(row["ReferenceFitting_AUROC_std"])
                else None
            )
            rf_auprc = (
                row["ReferenceFitting_AUPRC_mean"]
                if pd.notna(row["ReferenceFitting_AUPRC_mean"])
                else None
            )
            rf_auprc_std = (
                row["ReferenceFitting_AUPRC_std"]
                if pd.notna(row["ReferenceFitting_AUPRC_std"])
                else None
            )
            rf_time = (
                row["ReferenceFitting_training_time_mean"]
                if pd.notna(row["ReferenceFitting_training_time_mean"])
                else None
            )
            rf_time_std = (
                row["ReferenceFitting_training_time_std"]
                if pd.notna(row["ReferenceFitting_training_time_std"])
                else None
            )

            print(
                f"  N={system_size:3d}: StructureFlow AUROC={sf_auroc:.4f}±{sf_auroc_std:.4f}, AUPRC={sf_auprc:.4f}±{sf_auprc_std:.4f}, Time={sf_time:6.1f}±{sf_time_std:5.1f}s",
                end="",
            )
            if ngm_auroc is not None:
                print(
                    f" | NGM-NODE AUROC={ngm_auroc:.4f}±{ngm_auroc_std:.4f}, AUPRC={ngm_auprc:.4f}±{ngm_auprc_std:.4f}, Time={ngm_time:6.1f}±{ngm_time_std:5.1f}s",
                    end="",
                )
            else:
                print(" | NGM-NODE: N/A", end="")

            if rf_auroc is not None:
                print(
                    f" | RF AUROC={rf_auroc:.4f}±{rf_auroc_std:.4f}, AUPRC={rf_auprc:.4f}±{rf_auprc_std:.4f}, Time={rf_time:6.1f}±{rf_time_std:5.1f}s"
                )
            else:
                print(" | RF: N/A")


if __name__ == "__main__":
    # Create both versions of the plots
    print("Creating plots with confidence intervals...")
    fig_conf, stats_df = plot_scaling_results_with_confidence()

    print("\nCreating clean plots...")
    fig_clean, _ = plot_scaling_results_clean()

    # Print summary statistics
    plot_scaling_results_summary()
