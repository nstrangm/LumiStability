#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <filesystem>
#include <regex>
#include <tuple>
#include <fstream>
#include "/software/nstrangmann/CalorimeTree/Analysis/PlottingClass.h"

void checkFT0Shadowing(long trainNumber)
{

    std::string outputDir = "Data/Train_" + std::to_string(trainNumber);
    std::vector<int> runNumbers;

    // Check if directory exists
    if (!std::filesystem::exists(outputDir))
    {
        std::cout << "Directory " << outputDir << " does not exist!" << std::endl;
        return;
    }

    std::regex runPattern("^Run_(\\d{6})$"); // Regular expression to match Run_XXXXXX pattern (6 digits)

    // Iterate through directory entries
    for (const auto &entry : std::filesystem::directory_iterator(outputDir))
    {
        if (entry.is_directory())
        {
            std::string folderName = entry.path().filename().string();
            std::smatch match;

            if (std::regex_match(folderName, match, runPattern))
            {
                int runNumber = std::stoi(match[1].str());
                runNumbers.push_back(runNumber);
            }
        }
    }
    std::sort(runNumbers.begin(), runNumbers.end());

    Plotting1D PRR;
    std::vector<TH1F *> hMuChosenRatio;

    int ir = 0;

    for (const auto &runNumber : runNumbers)
    {

        std::vector<TString> triggers = {"FT0", "FDD"};

        for (const auto &trigger : triggers)
        {

            TFile *fTriggerRates = new TFile(Form("%s/Run_%d/TriggerRates.root", outputDir.c_str(), runNumber), "READ");
            TH1F *hMuSL;
            TH1F *hMuB;
            TH1F *hMuNSL;
            TH1F *hMuRatioSL;
            TH1F *hMuRatioNSL;
            TH1F *hMuRatioSLToNSL;

            PlottingRatio PR;
            PR.SetMargins(0.12, 0.12, 0.01, 0.01, 1500, 2000, 0.5);
            PR.NewRatioLine(0, 1, 3600, 1, 1, 1, kGray + 1);
            // Plotting1D PRO;

            TString triggerInString = trigger.Contains("FT0") ? "FT0VTx" : "FDD";
            TString SLInString = trigger.Contains("FT0") ? "T" : "D";
            TString triggerOutString = trigger.Contains("FT0") ? "TVX" : "FDD";
            float yLatexPos = trigger.Contains("FT0") ? 0.8 : 0.35;

            hMuSL = (TH1F *)fTriggerRates->Get(Form("%s_SLD100_MuVsBCID", triggerInString.Data()));
            hMuB = (TH1F *)fTriggerRates->Get(Form("%s_B_MuVsBCID", triggerInString.Data()));
            hMuNSL = (TH1F *)fTriggerRates->Get(Form("%s_NSLD100_MuVsBCID", triggerInString.Data()));

            hMuRatioSL = (TH1F *)hMuSL->Clone("hMuRatioSL");
            hMuRatioSL->Divide(hMuB);

            hMuRatioNSL = (TH1F *)hMuNSL->Clone("hMuRatioNSL");
            hMuRatioNSL->Divide(hMuB);

            hMuRatioSLToNSL = (TH1F *)hMuRatioSL->Clone("hMuRatioSLToNSL");
            hMuRatioSLToNSL->Divide(hMuRatioNSL);

            // PR.New(hMuB, "#mu_{B}", 21, 1, kBlue + 1);
            PR.New(hMuSL, "#mu_{SL}", 20, 1, kRed + 1);
            PR.New(hMuNSL, "#mu_{#bar{SL}}", 33, 1, kGreen + 1);
            // PR.NewRatio(hMuRatioSL, "#mu_{SL}/#mu_{B}", 20, 1, kRed + 1);
            // PR.NewRatio(hMuRatioNSL, "#mu_{#bar{SL}}/#mu_{B}", 33, 1, kGreen + 1);
            PR.NewRatio(hMuRatioSLToNSL, "", 34, 1, kMagenta + 1);

            float mean = 0;
            float stddev = 0;
            int nfilledBCs = 0;
            for (int i = 1; i <= hMuRatioSLToNSL->GetNbinsX(); i++)
            {
                float ratio = hMuRatioSLToNSL->GetBinContent(i);
                if (ratio == 0)
                    continue;
                nfilledBCs++;
                mean += hMuRatioSLToNSL->GetBinContent(i);
            }
            mean /= nfilledBCs;
            for (int i = 1; i <= hMuRatioSLToNSL->GetNbinsX(); i++)
            {
                float ratio = hMuRatioSLToNSL->GetBinContent(i);
                if (ratio == 0)
                    continue;
                stddev += (ratio - mean) * (ratio - mean);
            }
            stddev = std::sqrt(stddev / nfilledBCs);

            // std::cout << "Mean: " << mean << ", Stddev: " << stddev << std::endl;

            PR.NewRatioLatex(0.68, yLatexPos, Form("<#kern[0.2]{#mu_{SL}}/#mu_{#bar{SL}}> = %.3f", mean), StdTextSize);
            PR.NewRatioLatex(0.18, yLatexPos, Form("#bf{%s}", trigger.Data()), StdTextSize);

            PR.SetLegend(0.88, 0.95, 0.81, 0.96, true);
            PR.NewLatex(0.16, 0.81, Form("#bf{ALICE this thesis};Run %d (pp ref),#kern[0.8]{#mu} of %s trigger;SL definition: 100 preceding BCs w/o %s activity", runNumber, triggerOutString.Data(), trigger.Data()), StdTextSize, 0.1);
            PR.SetAxisRange(42, 42, 0.016, 0.074, 0.91, 1.19);
            PR.SetWhite(1E-6, 2E-6, 3E-6, 4E-6);
            PR.SetAxisLabel("#bf{Bunch crossing ID}", "#bf{#mu}", "#bf{#mu_{SL}/#mu_{#bar{SL}}}   ", 1., 1.45, 5);
            PR.Plot(Form("%s/Run_%d/Mu_%s.pdf", outputDir.c_str(), runNumber, trigger.Data()));
        }
    }
}