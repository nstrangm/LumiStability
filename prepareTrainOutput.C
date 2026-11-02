#include "DataFormatsParameters/GRPLHCIFData.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Configuration.h"
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <filesystem>
#include <regex>
#include <tuple>
#include <fstream>

enum Triggers
{
  FT0VTx = 0,
  FDD = 1,
  ZNC = 2,
  nTriggers = 3
};

TString triggers[Triggers::nTriggers] = {"FT0VTx", "FDD", "1ZNC"};
TString triggersOut[Triggers::nTriggers] = {"FT0", "FDD", "ZDC"};

enum BCType
{
  A = 0,
  B = 1,
  C = 2,
  E = 3,
  ABCE = 4,
  L = 5,
  LE = 6,
  SLD = 7,
  SLT = 8
};

const int nNEmptyBCVariations = 7;
const int NEmptyBCs[nNEmptyBCVariations] = {5, 10, 20, 50, 100, 200, 300};
const int nBCTypes = 5 + 4 * 7 * 2; // A + B + C + E + ABCE + (L + LE + SLT + SLD) * nNEmptyBCVariations * 2 (for the inverses)

TString BCSuffixIn[nBCTypes];
TString BCSuffixOut[nBCTypes];
TString BCFileSuffixIn[nBCTypes];
TString BCFileSuffixOut[nBCTypes];

void initializeBCTypes()
{
  // First set everything to ""
  for (int i = 0; i < nBCTypes; i++)
  {
    BCSuffixIn[i] = "";
    BCSuffixOut[i] = "";
    BCFileSuffixIn[i] = "";
    BCFileSuffixOut[i] = "";
  }
  // Now set A, B, C, E, ABCE
  BCSuffixIn[BCType::A] = "A";
  BCSuffixOut[BCType::A] = "A";
  BCSuffixIn[BCType::B] = "B";
  BCSuffixOut[BCType::B] = "B";
  BCSuffixIn[BCType::C] = "C";
  BCSuffixOut[BCType::C] = "C";
  BCSuffixIn[BCType::E] = "E";
  BCSuffixOut[BCType::E] = "E";
  BCSuffixIn[BCType::ABCE] = "ABCE";
  BCSuffixOut[BCType::ABCE] = "ABCE";

  const int nLBCDefs = 8;
  TString BCSuffixesIn[nLBCDefs] = {"L", "NL", "LE", "NLE", "SL_FDD", "NSL_FDD", "SL_FT0", "NSL_FT0"};
  TString BCSuffixesOut[nLBCDefs] = {"L", "NL", "LE", "NLE", "SLD", "NSLD", "SLT", "NSLT"};

  // Now set L, LE, SLT, SLD with variations for NEmptyBCs and their inverses
  int index = BCType::ABCE + 1;
  for (int i = 0; i < nNEmptyBCVariations; i++)
  {
    for (int j = 0; j < nLBCDefs; j++)
    {
      BCSuffixIn[index] = BCSuffixesIn[j];
      BCSuffixOut[index] = BCSuffixesOut[j];
      BCFileSuffixIn[index] = NEmptyBCs[i] == 5 ? "" : Form("_%03d", NEmptyBCs[i]);
      BCFileSuffixOut[index] = Form("%d", NEmptyBCs[i]);
      index++;
    }
  }

  for (int i = 0; i < nBCTypes; i++)
    std::cout << "BCType " << i << ": BCSuffixIn = " << BCSuffixIn[i].Data() << ", BCSuffixOut = " << BCSuffixOut[i].Data() << ", BCFileSuffixIn = " << BCFileSuffixIn[i].Data() << ", BCFileSuffixOut = " << BCFileSuffixOut[i].Data() << std::endl;
}

bool fillBCHistograms(TH1F *&hTriggersPerMinute, TH1F *&hBCIDs, TH1F *&hBCIDsInspected, int iTrigger, int iBCType, int runNumber, TFile *fAnalysisResults)
{
  hTriggersPerMinute = nullptr;
  hBCIDs = nullptr;
  hBCIDsInspected = nullptr;

  if (iBCType == BCType::ABCE)
  {
    TString path = Form("lumi-stability-light-ions%s/%s/BC_%s", BCFileSuffixIn[iBCType].Data(), triggers[iTrigger].Data(), "A");
    TH1F *hTime = (TH1F *)fAnalysisResults->Get(Form("%s/nBCsVsTime", path.Data()));
    TH1F *hBCID = (TH1F *)fAnalysisResults->Get(Form("%s/nBCsVsBCID", path.Data()));
    if (!hTime || !hBCID)
    {
      std::cout << "Warning: BC definition ABCE not found for trigger " << triggers[iTrigger].Data() << " in run " << runNumber << " - skipping" << std::endl;
      return false;
    }
    hTriggersPerMinute = (TH1F *)hTime->Clone(Form("hTriggersPerMinute_%s_%s_%s_Run%d", BCFileSuffixIn[iBCType].Data(), triggers[iTrigger].Data(), "A", runNumber));
    hBCIDs = (TH1F *)hBCID->Clone(Form("hTriggersPerBCId_%s_%s_%s_Run%d", BCFileSuffixIn[iBCType].Data(), triggers[iTrigger].Data(), "A", runNumber));
    hBCIDsInspected = (TH1F *)hBCID->Clone(Form("hTriggersPerBCIdInspected_%s_%s_%s_Run%d", BCFileSuffixIn[iBCType].Data(), triggers[iTrigger].Data(), "A", runNumber));
    std::vector<TString> BCs = {"B", "C", "E"};
    for (const auto &BC : BCs)
    {
      TString pathBC = Form("lumi-stability-light-ions%s/%s/BC_%s", BCFileSuffixIn[iBCType].Data(), triggers[iTrigger].Data(), BC.Data());
      TH1F *hTimeBC = (TH1F *)fAnalysisResults->Get(Form("%s/nBCsVsTime", pathBC.Data()));
      TH1F *hBCIDBC = (TH1F *)fAnalysisResults->Get(Form("%s/nBCsVsBCID", pathBC.Data()));
      TH1F *hBCIDInspectedBC = (TH1F *)fAnalysisResults->Get(Form("%s/nBCsInspectedVsBCID", pathBC.Data()));
      if (!hTimeBC || !hBCIDBC || !hBCIDInspectedBC)
      {
        std::cout << "Warning: BC_" << BC.Data() << " not found for ABCE / trigger " << triggers[iTrigger].Data() << " in run " << runNumber << " - skipping" << std::endl;
        delete hTriggersPerMinute;
        hTriggersPerMinute = nullptr;
        delete hBCIDs;
        hBCIDs = nullptr;
        delete hBCIDsInspected;
        hBCIDsInspected = nullptr;
        return false;
      }
      hTriggersPerMinute->Add(hTimeBC);
      hBCIDs->Add(hBCIDBC);
      hBCIDsInspected->Add(hBCIDInspectedBC);
    }
  }
  else
  {
    TString path = Form("lumi-stability-light-ions%s/%s/BC_%s", BCFileSuffixIn[iBCType].Data(), triggers[iTrigger].Data(), BCSuffixIn[iBCType].Data());
    TH1F *hTime = (TH1F *)fAnalysisResults->Get(Form("%s/nBCsVsTime", path.Data()));
    TH1F *hBCID = (TH1F *)fAnalysisResults->Get(Form("%s/nBCsVsBCID", path.Data()));
    TH1F *hBCIDInspected = (TH1F *)fAnalysisResults->Get(Form("lumi-stability-light-ions%s/AllBCs/BC_%s/nBCsInspectedVsBCID", BCFileSuffixIn[iBCType].Data(), BCSuffixIn[iBCType].Data()));
    // TH1F *hBCIDInspected = (TH1F *)fAnalysisResults->Get(Form("lumi-stability-light-ions%s/AllBCs/BC_%s/nBCsVsBCID", BCFileSuffixIn[iBCType].Data(), BCSuffixIn[iBCType].Data()));
    if (!hTime || !hBCID || !hBCIDInspected)
    {
      std::cout << "Warning: BC definition " << BCSuffixIn[iBCType].Data() << BCFileSuffixIn[iBCType].Data() << " not found for trigger " << triggers[iTrigger].Data() << " in run " << runNumber << " - skipping" << std::endl;
      return false;
    }
    hTriggersPerMinute = (TH1F *)hTime->Clone(Form("hTriggersPerMinute_%s_%s_%s_Run%d", BCFileSuffixIn[iBCType].Data(), triggers[iTrigger].Data(), BCSuffixOut[iBCType].Data(), runNumber));
    hBCIDs = (TH1F *)hBCID->Clone(Form("hTriggersPerBCId_%s_%s_%s_Run%d", BCFileSuffixIn[iBCType].Data(), triggers[iTrigger].Data(), BCSuffixOut[iBCType].Data(), runNumber));
    hBCIDsInspected = (TH1F *)hBCIDInspected->Clone(Form("hTriggersPerBCIdInspected_%s_%s_%s_Run%d", BCFileSuffixIn[iBCType].Data(), triggers[iTrigger].Data(), BCSuffixOut[iBCType].Data(), runNumber));
  }
  return true;
}

std::tuple<int, long, long, long> getRunDetails(int runNumber)
{
  // Get reference instead of copying
  auto &mCCDBManager = o2::ccdb::BasicCCDBManager::instance();
  const char *path = "http://alice-ccdb.cern.ch";
  mCCDBManager.setURL(path);

  auto runDuration = mCCDBManager.getRunDuration(runNumber);
  auto timestamp = runDuration.first + (runDuration.second - runDuration.first) / 2;
  mCCDBManager.setTimestamp(timestamp);
  std::map<std::string, std::string> metadata;
  metadata["runNumber"] = std::to_string(runNumber);

  const auto *mGRPECS = mCCDBManager.getSpecific<o2::parameters::GRPECSObject>("GLO/Config/GRPECS", timestamp, metadata);

  if (runNumber != mGRPECS->getRun())
    std::cout << "ERROR!!! Run from input and from GRPECS doesn't match! Run in: " << runNumber << " run GRP: " << mGRPECS->getRun() << std::endl;

  const auto *mGRPLHCIF = mCCDBManager.get<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF");
  auto fillTime = mGRPLHCIF->getFillNumberTime();
  auto fillNumber = mGRPLHCIF->getFillNumber();

  return std::make_tuple(fillNumber, fillTime, runDuration.first, runDuration.second);
}

void prepareTrainOutput(long trainNumber)
{
  initializeBCTypes();
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

  std::string csvPath = outputDir + "/Output.csv";
  std::ofstream csv(csvPath);
  csv << "timestamp,time in fill,fill_number,runNumber";

  for (int iT = 0; iT < Triggers::nTriggers; iT++)
  {
    for (int iB = 0; iB < nBCTypes; iB++)
    {
      csv << Form(",%s_%s%s", triggersOut[iT].Data(), BCSuffixOut[iB].Data(), BCFileSuffixOut[iB].Data());
    }
  }
  csv << "\n";

  for (const auto &runNumber : runNumbers)
  {
    auto [fillNumber, fillTime, runStart, runEnd] = getRunDetails(runNumber);
    std::cout << "Run " << runNumber << ", Fill: " << fillNumber << ", Fill Time: " << fillTime << ", Start: " << runStart << ", End: " << runEnd << std::endl;
    int runDurationMinutes = (runEnd - runStart) / 60000;
    int minBetweenSOFandSOR = (runStart - fillTime) / 60000;
    std::cout << "Run duration (minutes): " << runDurationMinutes << " minBeforeSOR = " << minBetweenSOFandSOR << std::endl;

    TFile *fAnalysisResults = new TFile(Form("%s/Run_%d/AnalysisResults.root", outputDir.c_str(), runNumber), "READ");
    TH1F *hTFsPerMinute = (TH1F *)fAnalysisResults->Get("lumi-stability-light-ions/TFsPerMinute");
    TH1F *hTriggersPerMinute[Triggers::nTriggers][nBCTypes];
    TH1F *hRate[Triggers::nTriggers][nBCTypes];
    TH1F *hCorrectedRate[Triggers::nTriggers][nBCTypes];
    TH1F *hMu[Triggers::nTriggers][nBCTypes];
    TH1F *hMuVsBCID[Triggers::nTriggers][nBCTypes];
    TH1F *hPileUpCorrection[Triggers::nTriggers][nBCTypes];
    TH1F *hBCIDs[Triggers::nTriggers][nBCTypes];
    TH1F *hBCIDInspected[Triggers::nTriggers][nBCTypes];
    int nFilledBCs[Triggers::nTriggers][nBCTypes];
    for (int iT = 0; iT < Triggers::nTriggers; iT++)
    {
      for (int iB = 0; iB < nBCTypes; iB++)
      {
        bool bcOk = fillBCHistograms(hTriggersPerMinute[iT][iB], hBCIDs[iT][iB], hBCIDInspected[iT][iB], iT, iB, runNumber, fAnalysisResults);
        if (!bcOk)
        {
          hRate[iT][iB] = nullptr;
          hCorrectedRate[iT][iB] = nullptr;
          hMu[iT][iB] = nullptr;
          hPileUpCorrection[iT][iB] = nullptr;
          nFilledBCs[iT][iB] = 0;
          continue;
        }
        hRate[iT][iB] = (TH1F *)hTriggersPerMinute[iT][iB]->Clone(Form("hRate%d%d_Run%d", iT, iB, runNumber));
        hCorrectedRate[iT][iB] = (TH1F *)hTriggersPerMinute[iT][iB]->Clone(Form("hCorrectedRate%d%d_Run%d", iT, iB, runNumber));

        hMu[iT][iB] = (TH1F *)hTriggersPerMinute[iT][iB]->Clone(Form("h%d%dMu_Run%d", iT, iB, runNumber));
        hPileUpCorrection[iT][iB] = (TH1F *)hTriggersPerMinute[iT][iB]->Clone(Form("h%d%dPileUpCorrection_Run%d", iT, iB, runNumber));
        for (int iBin = 1; iBin < hMu[iT][iB]->GetNbinsX() + 1; iBin++)
        {
          hMu[iT][iB]->SetBinContent(iBin, 0);
          hPileUpCorrection[iT][iB]->SetBinContent(iBin, 0);
        }

        hMuVsBCID[iT][iB] = (TH1F *)hBCIDs[iT][iB]->Clone(Form("h%d%dMuVsBCID_Run%d", iT, iB, runNumber));

        nFilledBCs[iT][iB] = 0;
        for (int iBin = 1; iBin < hBCIDs[iT][iB]->GetNbinsX() + 1; iBin++)
        {
          float nT = hBCIDs[iT][iB]->GetBinContent(iBin);
          if (nT > 0)
            nFilledBCs[iT][iB]++;
          float nI = hBCIDInspected[iT][iB]->GetBinContent(iBin);
          if (nI == 0)
            continue;
          float mu = -std::log(1. - nT / nI);
          hMuVsBCID[iT][iB]->SetBinContent(iBin, mu);
        }
      }
    }

    for (int iMinInRun = -1; iMinInRun < runDurationMinutes + 1; iMinInRun++)
    {
      int timeInFill = minBetweenSOFandSOR + iMinInRun;
      int nTFs = hTFsPerMinute->GetBinContent(timeInFill + 1);
      if (nTFs == 0)
        continue;            // Skip minutes with no TFs
      if (timeInFill > 1440) // Time histogram only supports fills of up to 24h - skip times after that
        continue;
      csv << fillTime + timeInFill * 60000 << "," << timeInFill << "," << fillNumber << "," << runNumber;
      for (int iT = 0; iT < Triggers::nTriggers; iT++)
      {
        for (int iB = 0; iB < nBCTypes; iB++)
        {
          if (!hTriggersPerMinute[iT][iB])
          {
            csv << ",";
            continue;
          }
          int nTriggersThisMinute = hTriggersPerMinute[iT][iB]->GetBinContent(timeInFill + 1);
          float rateHz = ((float)nTriggersThisMinute * 351.425 / (float)nTFs);
          float mu = -std::log(1. - rateHz / nFilledBCs[iT][iB] / o2::constants::lhc::LHCRevFreq);
          // cout << "Run " << runNumber << ", Trigger " << triggers[iT].Data() << ", BC Type " << BCSuffix[iB].Data() << ": nFilled BCs = " << nFilledBCs[iT][iB] << ", nTriggers = " << nTriggersThisMinute << ", Rate = " << rateHz << " Hz, Mu = " << mu << std::endl;
          hMu[iT][iB]->SetBinContent(timeInFill + 1, mu);
          float pileUpCorr = mu / (1. - std::exp(-mu));
          if (!(mu > 1E-5 && mu < 1)) // Avoid issues with very low mu where pile-up correction can be unstable
            pileUpCorr = 1.0;
          hPileUpCorrection[iT][iB]->SetBinContent(timeInFill + 1, pileUpCorr);
          csv << "," << rateHz * pileUpCorr;
          hRate[iT][iB]->SetBinContent(timeInFill + 1, rateHz);
          hCorrectedRate[iT][iB]->SetBinContent(timeInFill + 1, rateHz * pileUpCorr);
        }
      }
      csv << "\n";
    }
    TFile *fMuOut = new TFile(Form("%s/Run_%d/TriggerRates.root", outputDir.c_str(), runNumber), "RECREATE");
    for (int iT = 0; iT < Triggers::nTriggers; iT++)
    {
      for (int iB = 0; iB < nBCTypes; iB++)
      {
        if (!hTriggersPerMinute[iT][iB])
          continue;
        hTriggersPerMinute[iT][iB]->GetYaxis()->SetTitle("Triggers per minute");
        hTriggersPerMinute[iT][iB]->Write(Form("%s_%s%s_NTriggers", triggers[iT].Data(), BCSuffixOut[iB].Data(), BCFileSuffixOut[iB].Data()));
        hRate[iT][iB]->GetYaxis()->SetTitle("Rate (Hz)");
        hRate[iT][iB]->Write(Form("%s_%s%s_Rate", triggers[iT].Data(), BCSuffixOut[iB].Data(), BCFileSuffixOut[iB].Data()));
        hCorrectedRate[iT][iB]->GetYaxis()->SetTitle("Pile-up corrected Rate (Hz)");
        hCorrectedRate[iT][iB]->Write(Form("%s_%s%s_CorrectedRate", triggers[iT].Data(), BCSuffixOut[iB].Data(), BCFileSuffixOut[iB].Data()));
        hMu[iT][iB]->GetYaxis()->SetTitle("#mu");
        hMu[iT][iB]->Write(Form("%s_%s%s_Mu", triggers[iT].Data(), BCSuffixOut[iB].Data(), BCFileSuffixOut[iB].Data()));
        hPileUpCorrection[iT][iB]->GetYaxis()->SetTitle("Pile-up Correction");
        hPileUpCorrection[iT][iB]->Write(Form("%s_%s%s_PileUpCorrection", triggers[iT].Data(), BCSuffixOut[iB].Data(), BCFileSuffixOut[iB].Data()));
        hBCIDs[iT][iB]->Write(Form("%s_%s%s_BCIDs", triggers[iT].Data(), BCSuffixOut[iB].Data(), BCFileSuffixOut[iB].Data()));
        hBCIDInspected[iT][iB]->Write(Form("%s_%s%s_InspectedBCIDs", triggers[iT].Data(), BCSuffixOut[iB].Data(), BCFileSuffixOut[iB].Data()));
        hMuVsBCID[iT][iB]->Write(Form("%s_%s%s_MuVsBCID", triggers[iT].Data(), BCSuffixOut[iB].Data(), BCFileSuffixOut[iB].Data()));
      }
    }
    fMuOut->Close();
  }
  csv.close();
}