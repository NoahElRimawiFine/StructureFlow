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
    csv_data = """num_vars,seed,sparsity_target,sparsity_actual,true_edges,edge_density,max_eigenvalue_real,Correlation-pearson_AUROC,Correlation-pearson_AUPRC,Correlation-pearson_training_time,Correlation-pearson_num_true_edges,StructureFlow_AUROC,StructureFlow_AUPRC,StructureFlow_training_time,StructureFlow_num_true_edges,NGM-NODE_AUROC,NGM-NODE_AUPRC,NGM-NODE_training_time,NGM-NODE_num_true_edges,experiment_id,total_experiments
10,4852,0.05,0.06666666666666667,6,0.06666666666666667,0.0,0.9734042553191489,0.7028138528138528,0.018625736236572266,6,0.9627659574468085,0.509090909090909,57.84497809410095,6,0.7810283687943262,0.3880834899749373,322.7519874572754,6.0,1,36
10,4852,0.2,0.26666666666666666,24,0.26666666666666666,0.395655448780167,0.8135964912280701,0.5441361603475161,0.012085437774658203,24,0.7757675438596491,0.4692447182908081,57.832767486572266,24,0.7785087719298245,0.47540794937623915,324.6383876800537,24.0,2,36
10,4852,0.4,0.4111111111111111,37,0.4111111111111111,1.3182096866621533,0.8631488631488632,0.7907212834471284,0.012024879455566406,37,0.8421278421278421,0.7562902634055009,57.44078350067139,37,0.752895752895753,0.6563467965949508,328.1729917526245,37.0,3,36
25,4852,0.05,0.043333333333333335,26,0.043333333333333335,0.5922078331269707,0.9792603056376012,0.5439231326379298,0.06764602661132812,26,0.9780724284063181,0.5028348937550893,87.44798159599304,26,0.7698407602414281,0.17294465805593287,1063.5547759532928,26.0,4,36
25,4852,0.2,0.20333333333333334,122,0.20333333333333334,1.4305823939895372,0.8249193364403741,0.5595433355443513,0.06808209419250488,122,0.834493041749503,0.5292155372923445,77.5687415599823,122,0.7024655346608872,0.37147276335234236,997.8577566146851,122.0,5,36
25,4852,0.4,0.425,255,0.425,1.9448587728752682,0.7307154213036565,0.6671333191730175,0.06723880767822266,255,0.716120826709062,0.6356068852578105,77.48430800437927,255,0.5623635400105989,0.4655221440980293,824.9652671813965,255.0,6,36
50,4852,0.05,0.05061224489795919,124,0.05061224489795919,0.9029008473795888,0.9500753502769631,0.5227702410383083,0.44240498542785645,124,0.9247023324644292,0.47501918118317865,152.57684564590454,124,0.5977449223416965,0.0778863190985196,3123.374433517456,124.0,7,36
50,4852,0.2,0.20530612244897958,503,0.20530612244897958,2.3717790475785843,0.8656085519929995,0.6316451281948303,0.43462538719177246,503,0.812686226158323,0.5528893190794936,119.93507289886475,503,0.5411253062496328,0.23896093426619205,2550.32168841362,503.0,8,36
50,4852,0.4,0.4008163265306122,982,0.4008163265306122,2.9518286499428354,0.7080847883778904,0.6359765607805926,0.42153072357177734,982,0.642092916233977,0.5640048208068421,119.54320907592773,982,0.5557294140376581,0.4414136447103131,1409.912146806717,982.0,9,36
100,4852,0.05,0.05141414141414141,509,0.05141414141414141,1.5918627066326754,0.9705788898551185,0.5697944849005392,1.9889872074127197,509,0.9495254422605721,0.5183048428380559,237.41640973091125,509,0.5233350010629447,0.05770131366065983,10360.619023323059,509.0,10,36
100,4852,0.2,0.2006060606060606,1986,0.2006060606060606,3.2186210953942624,0.8243974354044572,0.5559764380525823,2.2313003540039062,1986,0.7595058659933234,0.4685942391145377,231.6057665348053,1986,0.5109463210278287,0.2035805347143642,3091.8110506534576,1986.0,11,36
100,4852,0.4,0.39616161616161616,3922,0.39616161616161616,4.323132007151699,0.6578295686586025,0.5801396233834728,4.368826866149902,3922,0.6195861248944748,0.5296250846965184,233.59105587005615,3922,0.5092083552941457,0.3973237266569635,2388.5191402435303,3922.0,12,36
200,4852,0.05,0.05050251256281407,2010,0.05050251256281407,2.2949998574409403,0.96814073879091,0.5526191264144195,7.927350759506226,2010,0.6165340774411701,0.21256994183718592,571.4490671157837,2010,,,,,13,36
200,4852,0.2,0.2014070351758794,8016,0.2014070351758794,5.244219711505882,0.7674918500293061,0.4897851792773821,7.621973991394043,8016,0.6868925843441931,0.38349873985718347,816.0332379341125,8016,,,,,14,36
200,4852,0.4,0.4037437185929648,16069,0.4037437185929648,6.501912281236235,0.6218929139158376,0.5434334918619736,9.272216081619263,16069,0.5659527723198657,0.4721297261061206,733.2379455566406,16069,,,,,15,36
500,4852,0.05,0.04988777555110221,12447,0.04988777555110221,3.8611548089595615,0.9566752835561358,0.5056815473923015,68.93512034416199,12447,0.630302922060287,0.15696211134066831,5138.624960422516,12447,,,,,16,36
500,4852,0.2,0.19991583166332666,49879,0.19991583166332666,7.902942738804681,0.6907708196953042,0.39733471936704684,60.267064571380615,49879,0.5289162359044275,0.22013346016343596,5084.449959278107,49879,,,,,17,36
500,4852,0.4,0.4006733466933868,99968,0.4006733466933868,10.794718762436169,0.5796870370572546,0.4911130286332667,62.461748361587524,99968,0.5104833479061066,0.4090570671710845,5428.748547077179,99968,,,,,18,36
10,2502,0.05,0.03333333333333333,3,0.03333333333333333,0.0,0.993127147766323,0.8055555555555556,0.24680805206298828,3,0.9828178694158075,0.6111111111111112,60.89986515045166,3,0.5532646048109965,0.055272108843537414,313.9014484882355,3.0,19,36
10,2502,0.2,0.14444444444444443,13,0.14444444444444443,0.45274256050913675,0.9336870026525199,0.6670541199116324,0.021987438201904297,13,0.9332449160035367,0.6064580347189044,57.0928475856781,13,0.7780725022104332,0.43960817162269067,328.5199406147003,13.0,20,36
10,2502,0.4,0.3333333333333333,30,0.3333333333333333,0.8311741245831212,0.8638095238095238,0.7129846759193442,0.011591434478759766,30,0.8390476190476192,0.6508294299766001,56.757171869277954,30,0.7552380952380953,0.5807710817767366,336.25325417518616,30.0,21,36
25,2502,0.05,0.05333333333333334,32,0.05333333333333334,0.7196140462937763,0.9381323777403034,0.5174637896473324,0.0668649673461914,32,0.9517811973018548,0.5160586468732333,80.30390334129333,32,0.692295531197302,0.23074941235248975,1068.1875479221344,32.0,22,36
25,2502,0.2,0.19833333333333333,119,0.19833333333333333,1.3123505169597558,0.8755272860132195,0.5792912693317145,0.06846141815185547,119,0.8422792041717873,0.5339294313329173,78.74769520759583,119,0.6186601122662504,0.3039079820751982,1081.0779702663422,119.0,23,36
25,2502,0.4,0.385,231,0.385,1.732993189034112,0.7258663502318325,0.6388499507982699,0.06640362739562988,231,0.7234766079943745,0.6233801558082154,79.93856620788574,231,0.6033632188454524,0.4766637796190075,995.7301144599915,231.0,24,36
50,2502,0.05,0.05061224489795919,124,0.05061224489795919,1.011096206290654,0.9766515966112741,0.570391982943973,0.74066162109375,124,0.9628102259150646,0.5027629030484266,128.71913838386536,124,0.6237271912675139,0.09157567784636376,3192.3506700992584,124.0,25,36
50,2502,0.2,0.20489795918367346,502,0.20489795918367346,2.2483800453103733,0.8555806802818755,0.5841160314833563,0.36829495429992676,502,0.8002683958859258,0.5185753101194658,121.33024144172668,502,0.5446083533732936,0.2342731467410447,2368.58771276474,502.0,26,36
50,2502,0.4,0.4048979591836735,992,0.4048979591836735,2.864936135818015,0.704673863052965,0.6335978337504595,0.541236162185669,992,0.6435709816462736,0.5677720499751625,119.8461434841156,992,0.5446442895952769,0.4405993383653992,1217.2671146392822,992.0,27,36
100,2502,0.05,0.05171717171717172,512,0.05171717171717172,1.621375368082806,0.9735850548060708,0.5677566445259788,1.9422035217285156,512,0.954265214942559,0.5215528299995554,231.66685247421265,512,0.5519869259195826,0.06354449333459561,10017.93087887764,512.0,28,36
100,2502,0.2,0.20656565656565656,2045,0.20656565656565656,3.2842529847499775,0.8198650415924539,0.5568001360244068,2.343620777130127,2045,0.744290976596657,0.4645960126865602,232.7276487350464,2045,0.5122506027947547,0.2101899865277203,3071.490483045578,2045.0,29,36
100,2502,0.4,0.40545454545454546,4014,0.40545454545454546,4.275314334045096,0.6645226505093849,0.5940594166285376,1.9593420028686523,4014,0.6194301609918242,0.5382326014982242,234.22895073890686,4014,0.5131996457104444,0.41053888331431937,2378.5854711532593,4014.0,30,36
200,2502,0.05,0.051457286432160805,2048,0.051457286432160805,2.4913927807530074,0.9694603056368901,0.549223169181854,7.6718058586120605,2048,0.6050699115000526,0.19588061767962656,571.2099781036377,2048,,,,,31,36
200,2502,0.2,0.19924623115577889,7930,0.19924623115577889,4.6654139694379095,0.7738436412151697,0.490576798241162,7.632061004638672,7930,0.6866895221715108,0.379842272121609,571.0319414138794,7930,,,,,32,36
200,2502,0.4,0.39977386934673365,15911,0.39977386934673365,6.696018599710984,0.6306827180548562,0.5518513860566123,7.34396767616272,15911,0.574681301138012,0.47429580458434983,571.5179307460785,15911,,,,,33,36
500,2502,0.05,0.050569138276553106,12617,0.050569138276553106,3.7046298903105104,0.9528146534159547,0.49974282880661985,62.05327916145325,12617,0.6193330312429121,0.14535419062724012,5360.002261638641,12617,,,,,34,36
500,2502,0.2,0.2014188376753507,50254,0.2014188376753507,8.180315170301121,0.6895586017834803,0.3970475162183286,61.09390330314636,50254,0.5291851691465888,0.2208703164645075,5353.444791316986,50254,,,,,35,36
500,2502,0.4,0.40139078156312624,100147,0.40139078156312624,11.074895782446818,0.5797805580917403,0.49229326803046286,60.75722599029541,100147,0.5108339833222751,0.4111183272903037,5351.249730587006,100147,,,,,36,36"""

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

    # Set up the plotting style
    plt.style.use("default")

    # Define colors and line styles
    colors = {"StructureFlow": "#1f77b4", "NGM-NODE": "#ff7f0e"}
    line_styles = {0.05: "-", 0.2: "--", 0.4: ":"}
    sparsity_labels = {0.05: "5% sparse", 0.2: "20% sparse", 0.4: "40% sparse"}

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: AUROC vs System Size
    for sparsity in [0.05, 0.2, 0.4]:
        sparsity_data = stats_df[stats_df["sparsity"] == sparsity]

        # StructureFlow
        x_sf = sparsity_data["system_size"]
        y_sf = sparsity_data["StructureFlow_AUROC_mean"]
        y_err_sf = sparsity_data["StructureFlow_AUROC_std"] / np.sqrt(
            sparsity_data["StructureFlow_AUROC_count"]
        )
        ax1.errorbar(
            x_sf,
            y_sf,
            yerr=y_err_sf,
            color=colors["StructureFlow"],
            linestyle=line_styles[sparsity],
            marker="o",
            linewidth=2,
            markersize=6,
            capsize=3,
            capthick=1,
            label=f"StructureFlow ({sparsity_labels[sparsity]})",
        )

        # NGM-NODE (filter out NaN values)
        ngm_data = sparsity_data.dropna(subset=["NGM-NODE_AUROC_mean"])
        x_ngm = ngm_data["system_size"]
        y_ngm = ngm_data["NGM-NODE_AUROC_mean"]
        y_err_ngm = ngm_data["NGM-NODE_AUROC_std"] / np.sqrt(
            ngm_data["NGM-NODE_AUROC_count"]
        )
        ax1.errorbar(
            x_ngm,
            y_ngm,
            yerr=y_err_ngm,
            color=colors["NGM-NODE"],
            linestyle=line_styles[sparsity],
            marker="s",
            linewidth=2,
            markersize=6,
            capsize=3,
            capthick=1,
            label=f"NGM NeuralODE ({sparsity_labels[sparsity]})",
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

        # StructureFlow
        x_sf = sparsity_data["system_size"]
        y_sf = sparsity_data["StructureFlow_AUPRC_mean"]
        y_err_sf = sparsity_data["StructureFlow_AUPRC_std"] / np.sqrt(
            sparsity_data["StructureFlow_AUPRC_count"]
        )
        ax2.errorbar(
            x_sf,
            y_sf,
            yerr=y_err_sf,
            color=colors["StructureFlow"],
            linestyle=line_styles[sparsity],
            marker="o",
            linewidth=2,
            markersize=6,
            capsize=3,
            capthick=1,
            label=f"StructureFlow ({sparsity_labels[sparsity]})",
        )

        # NGM-NODE (filter out NaN values)
        ngm_data = sparsity_data.dropna(subset=["NGM-NODE_AUPRC_mean"])
        x_ngm = ngm_data["system_size"]
        y_ngm = ngm_data["NGM-NODE_AUPRC_mean"]
        y_err_ngm = ngm_data["NGM-NODE_AUPRC_std"] / np.sqrt(
            ngm_data["NGM-NODE_AUPRC_count"]
        )
        ax2.errorbar(
            x_ngm,
            y_ngm,
            yerr=y_err_ngm,
            color=colors["NGM-NODE"],
            linestyle=line_styles[sparsity],
            marker="s",
            linewidth=2,
            markersize=6,
            capsize=3,
            capthick=1,
            label=f"NGM NeuralODE ({sparsity_labels[sparsity]})",
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
    # Calculate averages for StructureFlow
    sf_avg_data = (
        stats_df.groupby("system_size")["StructureFlow_training_time_mean"]
        .mean()
        .reset_index()
    )
    ax3.plot(
        sf_avg_data["system_size"],
        sf_avg_data["StructureFlow_training_time_mean"],
        color=colors["StructureFlow"],
        linestyle="-",
        marker="o",
        linewidth=2,
        markersize=8,
        label="StructureFlow (avg across sparsities)",
    )

    # Calculate averages for NGM-NODE (only where data exists)
    ngm_avg_data = (
        stats_df.dropna(subset=["NGM-NODE_training_time_mean"])
        .groupby("system_size")["NGM-NODE_training_time_mean"]
        .mean()
        .reset_index()
    )
    ax3.plot(
        ngm_avg_data["system_size"],
        ngm_avg_data["NGM-NODE_training_time_mean"],
        color=colors["NGM-NODE"],
        linestyle="-",
        marker="s",
        linewidth=2,
        markersize=8,
        label="NGM NeuralODE (avg across sparsities)",
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

    # Adjust layout
    plt.tight_layout()

    # Save the plots
    plt.savefig("scaling_experiment_plots.png", dpi=300, bbox_inches="tight")
    print("Plots saved to scaling_experiment_plots.png")

    # Show summary statistics
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

            print(
                f"  N={system_size:3d}: StructureFlow AUROC={sf_auroc:.4f}±{sf_auroc_std:.4f}, AUPRC={sf_auprc:.4f}±{sf_auprc_std:.4f}, Time={sf_time:6.1f}±{sf_time_std:5.1f}s",
                end="",
            )
            if ngm_auroc is not None:
                print(
                    f" | NGM-NODE AUROC={ngm_auroc:.4f}±{ngm_auroc_std:.4f}, AUPRC={ngm_auprc:.4f}±{ngm_auprc_std:.4f}, Time={ngm_time:6.1f}±{ngm_time_std:5.1f}s"
                )
            else:
                print(" | NGM-NODE: N/A")

    # Show the plots
    plt.show()

    return fig, stats_df


def plot_auprc_comparison():
    """
    Note: AUPRC data not available in the new dataset.
    This function is kept for compatibility but will show a message.
    """
    print("AUPRC data not available in the new dataset.")
    return None


if __name__ == "__main__":
    # Create the main plots
    fig, stats_df = plot_scaling_results()

    # AUPRC comparison not available with new data
    plot_auprc_comparison()
