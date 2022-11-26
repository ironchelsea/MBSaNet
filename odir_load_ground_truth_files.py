import csv


class GroundTruthFiles:
    def __init__(self):
        self.amd = []
        self.cataract = []
        self.diabetes = []
        self.glaucoma = []
        self.hypertension = []
        self.myopia = []
        self.others = []

    def populate_vectors(self, ground_truth_file):
        with open(ground_truth_file) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)

            for row in csv_reader:
                column_id = row[0]
                normal = row[1]
                diabetes = row[2]
                glaucoma = row[3]
                cataract = row[4]
                amd = row[5]
                hypertension = row[6]
                myopia = row[7]
                others = row[8]
                # just discard the first row
                if column_id != "ID":
                    print("Processing image: " + column_id + "_left.jpg")
                    if diabetes == '1':
                        self.diabetes.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if glaucoma == '1':
                        self.glaucoma.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if cataract == '1':
                        self.cataract.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if amd == '1':
                        self.amd.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if hypertension == '1':
                        self.hypertension.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if myopia == '1':
                        self.myopia.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])
                    if others == '1':
                        self.others.append([column_id, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others])