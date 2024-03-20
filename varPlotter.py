import ROOT

enVars = ROOT.TFile.Open("boostedHTT_mt_2018_tauEn.input.root")
metVars = ROOT.TFile.Open("boostedHTT_mt_2018_metEn.input.root")
jetVars = ROOT.TFile.Open("boostedHTT_mt_2018_jetEn.input.root")


enSysts = ["_nominal", "_en_scale_up", "_en_scale_down"] 
metSysts = ["_MissingEn_JESUp", "_MissingEn_JESDown", "_MissingEn_UESUp", "_MissingEn_UESDown"]
jetSysts = ["_JESUp", "_JESDown"]
allNames = ["DYJets125", "TT", "WJets", "VV", "QCD"]
d = {}

for i in allNames:
    for j in enSysts:
        d["DYJets_met_1_13TeV/" + i + "" + j] = enVars.Get("DYJets_met_1_13TeV/" + i + "" + j)
    for j in metSysts:
        d["DYJets_met_1_13TeV/" + i + "" + j] = metVars.Get("DYJets_met_1_13TeV/" + i + "" + j)
    for j in jetSysts:
        d["DYJets_met_1_13TeV/" + i + "" + j] = jetVars.Get("DYJets_met_1_13TeV/" + i + "" + j)
print("Past name loop")


allSysts = ["_nominal", "_en_scale_up", "_en_scale_down", "_MissingEn_JESUp", "_MissingEn_JESDown", "_MissingEn_UESUp", "_MissingEn_UESDown", "_JESUp", "_JESDown"] 


fullFile = ROOT.TFile.Open("boostedHTT_mt_2018.input.root", "RECREATE")
direc = fullFile.mkdir("DYJets_met_1_13TeV")
fullFile.cd("DYJets_met_1_13TeV")
for i in allSysts:
    for j in allNames:
        ROOT.gDirectory.WriteObject(d["DYJets_met_1_13TeV/" + j + "" + i], "" + j + "" + i)
print("Done looping!")

allVars = ["Boosted_Tau_Energy_Scale", "Jet_Energy_Scale", "MET", "Unclustered_Energies"]
enSysts = ["_en_scale_up", "_en_scale_down"] 
metSysts = ["_MissingEn_JESUp", "_MissingEn_JESDown"] 
USysts = ["_MissingEn_UESUp", "_MissingEn_UESDown"]
jetSysts = ["_JESUp", "_JESDown"]


h1 = enVars.Get("DYJets_met_1_13TeV/DYJets125_nominal")
h1.SetMarkerStyle(20)
h1.SetMarkerColorAlpha(1, 1)
h1.SetLineColor(1)
h1.SetStats(0);
h1.GetXaxis().SetRangeUser(25,110)
h1.Sumw2(1)
c = ROOT.TCanvas("c", "canvas", 800, 800)
for i in allVars:
    c.Clear()
    if i == "Boosted_Tau_Energy_Scale":
        systs = enSysts
    if i == "Jet_Energy_Scale":
        systs = jetSysts
    if i == "MET":
        systs = metSysts
    if i == "Unclustered_Energies":
        systs = USysts

    h2 = fullFile.Get("DYJets_met_1_13TeV/DYJets125" + systs[0])
    h2.SetMarkerStyle(20)
    h2.GetYaxis().SetTitleSize(20)
    h2.GetYaxis().SetTitleFont(43)
    h2.GetYaxis().SetTitleOffset(1.55)
    h2.SetMarkerColorAlpha(2, 1)
    h2.SetLineColor(2)
    h2.GetXaxis().SetRangeUser(25,110)
    h2.SetStats(0);
    h2.Sumw2()


    h3 = fullFile.Get("DYJets_met_1_13TeV/DYJets125" + systs[1])
    h3.SetMarkerStyle(20)
    h3.GetYaxis().SetTitleSize(20)
    h3.GetYaxis().SetTitleFont(43)
    h3.GetYaxis().SetTitleOffset(1.55)
    h3.SetMarkerColorAlpha(4, 1)
    h3.SetLineColor(4)
    h3.GetXaxis().SetRangeUser(25,110)
    h3.SetStats(0);
    h3.Sumw2()


    h4 = h1.Clone("h4")
    h4.SetMarkerStyle(20)
    h4.SetMarkerColorAlpha(2, 1)
    h4.SetLineColor(2)
    h4.SetTitle("")
    h4.SetMinimum(0.8)
    h4.SetMaximum(1.35)
    h4.SetStats(0);
    h4.GetXaxis().SetRangeUser(25,110)
    h4.Sumw2()
    h4.Divide(h2)

    h5 = h1.Clone("h5")
    h5.SetMarkerStyle(20)
    h5.SetMarkerColorAlpha(4, 1)
    h5.SetLineColor(4)
    h5.SetTitle("")
    h5.SetMinimum(0.8)
    h5.SetMaximum(1.35)
    h5.SetStats(0);
    h5.GetXaxis().SetRangeUser(25,110)
    h5.Sumw2()
    h5.Divide(h3)




    c.cd(0)
    rp = ROOT.TRatioPlot(h1, h2, "divsym")
    rp.Draw()
    rp.GetUpperPad().cd()
    h1.Draw("SAME")
    h2.Draw("SAME")
    h3.Draw("SAME")
    rp.GetLowerPad().cd()
    rp.GetLowerRefGraph().SetMinimum(0)
    rp.GetLowerRefGraph().SetMaximum(2)
    rp.GetUpperRefObject().SetMaximum(50)
    h4.Draw("SAME")
    h5.Draw("SAME")
    c.cd(0)
    pad5 = ROOT.TPad("all","all",0,0,1,1);
    pad5.SetFillStyle(4000); 
    pad5.Draw("SAME");
    pad5.cd();
    lat = ROOT.TLatex();
    lat.DrawLatexNDC(.1,.95,"Systematic: " + i);
    c.SaveAs("./syst_plots/varRatio" + i + ".png")




