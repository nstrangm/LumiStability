#include "/software/nstrangmann/CalorimeTree/Analysis/PlottingClass.h"

void PlotStability(TString inputDir, TString luminometer1, TString luminometer2, TString BCType, TString plotLabel, TString lumiLuminometer = "FT0_B", TString refLumiName = "FDD")
{
    ENTER

    TFile *f = new TFile(Form("/alf/data/nstrangmann/LumiStability/%s/luminosity_histogram.root", inputDir.Data()));

    if (!f || f->IsZombie())
        return;

    std::cout << "Trying to get " << Form("lumi_%s", luminometer1.Data()) << " and " << Form("lumi_%s", luminometer2.Data()) << " from " << Form("/alf/data/nstrangmann/LumiStability/%s/luminosity_histogram.root", inputDir.Data()) << std::endl;
    TH1F *lT = (TH1F *)f->Get(Form("lumi_%s", luminometer1.Data()))->Clone("lT");
    TH1F *lZ = (TH1F *)f->Get(Form("lumi_%s", luminometer2.Data()))->Clone("lF");

    TH1F *lB = (TH1F *)f->Get(Form("lumi_%s", lumiLuminometer.Data()))->Clone("lB");

    TH1F *lR = new TH1F("lR", "lR", 100, 0.96, 1.04);
    for (int i = 1; i <= lZ->GetNbinsX(); i++)
    {
        double z = lZ->GetBinContent(i);
        double t = lT->GetBinContent(i);
        double lumi = lB->GetBinContent(i) * 60. / 1000.;
        double time = lZ->GetBinCenter(i);
        if (z > 0)
            lR->Fill(z / t, lumi);
    }

    for (int iBin = 0; iBin < lR->GetNbinsX(); iBin++)
        lR->SetBinError(iBin + 1, 0.0);

    float sigma = lR->GetStdDev();
    float mean = lR->GetMean();
    float unc = std::sqrt(sigma * sigma + (mean - 1) * (mean - 1));

    bool lumiInpb = false;
    float maxLumi = lR->GetMaximum();
    if (maxLumi > 1000){
        lumiInpb = true;
        lR->Scale(1. / 1000.);
    }

    float intLumi = lR->Integral();
    LOG(Form("Integrated luminosity in %s: %.2f %sb^{-1}", inputDir.Data(), intLumi, lumiInpb ? "p" : "n"))

    Plotting1D PR;
    PR.SetMargins(0.15, 0.12, 0.02, 0.02, 2000, 1200);
    // PR.SetMargins(0.12, 0.1, 0.05, 0.01,2500,1500);
    PR.New(lR, "", 1, 2, kBlack, "h");
    PR.SetAxisLabel(Form("#bf{#it{L}_{#kern[0.05]{%s}}#kern[0.3]{/}#it{L}_{#kern[0.05]{TVX}}}", refLumiName.Data()), Form("#bf{#it{L}_{#kern[0.05]{TVX}} (%sb^{-1})}", lumiInpb ? "p" : "n"), 1., 1.5);
    // float intLumi = lR->Integral();
    // bool lumiInpb = intLumi > 1000;
    // if (lumiInpb)
    //     intLumi /= 1000.;
    PR.NewLatex(0.16, 0.89, Form("#bf{ALICE this thesis};%s;%s;#bar{r} = %.3f, #sigma = %.3f;#sqrt{(#bar{r}-1)^{2}+#sigma^{2}} = %.2f%%", plotLabel.Data(), BCType.Data(), mean, sigma, unc * 100), StdTextSize, 0.07);
    // PR.NewLatex(0.15, 0.88, Form("#bf{ALICE this thesis};%s;#it{L}_{TVX}^{int} = %.2f %sb^{-1};%s;#mu = %.3f, #sigma = %.3f; #sqrt{(#mu-1)^{2}+#sigma^{2}} = %.2f%%", plotLabel.Data(), intLumi, lumiInpb ? "p" : "n", BCType.Data(), mean, sigma, unc * 100));
    PR.SetAxisRange(0.971, 1.029, 0.);
    PR.Plot(Form("%s/LuminosityRatio_%s%s.pdf", inputDir.Data(), inputDir.Data(), refLumiName == "FDD" ? "" : "ZDC"));

    EXIT
}

void plotStabilities()
{
    ENTER

    PlotStability("pp_23", "FT0_L100", "FDD_L100", "Leading 100 BCs", "pp,#kern[0.3]{#sqrt{#it{s}}}#kern[0.5]{=}#kern[0.2]{13}.6#kern[0.3]{T}eV (2023)");
    PlotStability("pp_24", "FT0_SLD100", "FDD_SLD100", "Super Leading 100 BCs", "pp,#kern[0.3]{#sqrt{#it{s}}}#kern[0.5]{=}#kern[0.2]{13}.6#kern[0.3]{T}eV (2024)");
    PlotStability("pp_25", "FT0_SLD100", "FDD_SLD100", "Super Leading 100 BCs", "pp,#kern[0.3]{#sqrt{#it{s}}}#kern[0.5]{=}#kern[0.2]{13}.6#kern[0.3]{T}eV (2025)");
    PlotStability("pp_26", "FT0_SLD100", "FDD_SLD100", "Super Leading 100 BCs", "pp,#kern[0.3]{#sqrt{#it{s}}}#kern[0.5]{=}#kern[0.2]{13}.6#kern[0.3]{T}eV (2026)");
    PlotStability("OO", "FT0_B", "FDD_B", "Any BC", "OO,#kern[0.2]{#sqrt{#it{s}_{#scale[0.8]{NN}}}}#kern[0.5]{=}#kern[0.5]{5}.36#kern[0.3]{T}eV", "FT0_B");
    PlotStability("OO", "FT0_B", "ZDC_B", "Any BC", "OO,#kern[0.2]{#sqrt{#it{s}_{#scale[0.8]{NN}}}}#kern[0.5]{=}#kern[0.5]{5}.36#kern[0.3]{T}eV", "FT0_B", "1ZNC");
    PlotStability("pO", "FT0_B", "FDD_B", "Any BC", "pO,#kern[0.2]{#sqrt{#it{s}_{#scale[0.8]{NN}}}}#kern[0.5]{=}#kern[0.5]{9}.62#kern[0.3]{T}eV", "FT0_B");
    PlotStability("pO", "FT0_B", "ZDC_B", "Any BC", "pO,#kern[0.2]{#sqrt{#it{s}_{#scale[0.8]{NN}}}}#kern[0.5]{=}#kern[0.5]{9}.62#kern[0.3]{T}eV", "FT0_B", "1ZNA");
    PlotStability("pp_ref", "FT0_SLD100", "FDD_SLD100", "Super Leading 100 BCs", "pp,#kern[0.3]{#sqrt{#it{s}}}#kern[0.5]{=}#kern[0.5]{5}.36#kern[0.3]{T}eV");

    EXIT
}