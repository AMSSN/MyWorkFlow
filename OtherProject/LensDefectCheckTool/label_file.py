from utils import *


class LabelFile(object):
    suffix = ".json"
    empty_check_dict = {"manu_defects": [], "remarks": [""]}

    def __init__(self, targetFileDir):
        self.targetFileDir = targetFileDir
        self.manu_check_result = {}
        self.json_path = osp.join(self.targetFileDir, "manu_check_remark.json")

    def loadManuCheckResult(self):
        if os.path.exists(self.json_path):
            fr = open(self.json_path, "r", encoding="utf-8")
            self.manu_check_result = json.load(fr)
            fr.close()

    def updateCheckIdDict(self, id_direct, check_edit_dict):
        if id_direct is not None:
            if id_direct in self.manu_check_result.keys():
                check_raw_dict = self.manu_check_result[id_direct]
            else:
                check_raw_dict = self.empty_check_dict

            if check_edit_dict != check_raw_dict:
                self.manu_check_result.update({id_direct: check_edit_dict})
                if check_edit_dict == self.empty_check_dict:
                    del self.manu_check_result[id_direct]
                print("write json!!!")
                json_str = json.dumps(
                    self.manu_check_result, indent=4, ensure_ascii=False
                )
                fw = open(self.json_path, "w", encoding="utf-8")
                fw.write(json_str)
                fw.close()

    def getCheckIdDict(self, id_direct):
        if id_direct in self.manu_check_result.keys():
            check_raw_dict = self.manu_check_result[id_direct]
        else:
            check_raw_dict = self.empty_check_dict
        return check_raw_dict

    def judgeCheckDictHasMesg(self, check_edit_dict):
        if check_edit_dict == self.empty_check_dict:
            return False
        else:
            return True
