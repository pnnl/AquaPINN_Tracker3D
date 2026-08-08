import numpy as np
import AML_solver

speed_of_light = 299792458  # meters per second

class TDOAGroup:
    def __init__(self, tdoa_group, pos_rx):
        self.tdoa_group = tdoa_group
        self.pos_rx     = pos_rx
        self.solved_loc = None  # Initialize solved_loc attribute
        self.receiver_list = self.get_receiver_list() 
        self.tdoa_matrix   = self.get_tdoa_matrix()
        self.pos_rx_used = {key: self.pos_rx[key] for key in self.receiver_list if key in self.pos_rx}


    def print_tdoa_info_group_info(self):
        print("Group ID:",      self.tdoa_group.group_id, end= "; ")
        print("Timestamp:",     self.tdoa_group.timestamp, end= "; ")
        print("Transmitter:",   self.tdoa_group.tx)
        for i, tdoa_data in enumerate(self.tdoa_group.tdoas):
            print(f"\tTDOA Data {i+1}:", end= " ")
            print("Receiver Pair: (", tdoa_data["rx0"], ",", tdoa_data["rx1"], ")", end= "; ")
            print("TDOA:",  tdoa_data["tdoa"])
            print("\tSNR:", tdoa_data["snr"], end= "; ")
            print("Model Quality:",     tdoa_data["model_quality"], end= "; ")
            print("Detection Pair: (",  tdoa_data["det0_idx"], ",",tdoa_data["det1_idx"], ")")


    def get_receiver_list(self):
        return np.unique([f["rx0"] for f in self.tdoa_group.tdoas] 
                         + [f["rx1"] for f in self.tdoa_group.tdoas])
    

    ## more efficient code to get tdoa matrix
    def get_tdoa_matrix(self):
        num_receivers = len(self.receiver_list)
        tdoa_dict = {}

        # Populate the dictionary with TDOA values for each pair of receivers
        for tdoa_data in self.tdoa_group.tdoas:
            rx0, rx1 = tdoa_data["rx0"], tdoa_data["rx1"]
            tdoa = tdoa_data["tdoa"]
            if (rx0, rx1) in tdoa_dict:
                tdoa_dict[(rx0, rx1)].append(tdoa)
            else:
                tdoa_dict[(rx0, rx1)] = [tdoa]
            if (rx1, rx0) in tdoa_dict:
                tdoa_dict[(rx1, rx0)].append(-tdoa)
            else:
                tdoa_dict[(rx1, rx0)] = [-tdoa]

        # Create the TDOA matrix using the dictionary
        tdoa_mat = np.zeros((num_receivers, num_receivers))
        for i, ref_rx in enumerate(self.receiver_list):
            for j, rx_id in enumerate(self.receiver_list):
                tdoa_values_forward = tdoa_dict.get((ref_rx, rx_id), [])
                tdoa_values_reverse = tdoa_dict.get((rx_id, ref_rx), [])
                tdoa_values = tdoa_values_forward + tdoa_values_reverse
                if tdoa_values:
                    tdoa_mat[i][j] = tdoa_values[0] * 1e9  # Store the first value of the list

        return tdoa_mat


    # def get_tdoa_matrix(self):
    #     num_receivers = len(self.receiver_list)
    #     tdoa_mat = np.zeros((num_receivers, num_receivers))
    #     for i, ref_rx in enumerate(self.receiver_list):
    #         for j, rx_id in enumerate(self.receiver_list):
    #             tdoa_values_direct = [f["tdoa"] for f in self.tdoa_group.tdoas 
    #                                 if (ref_rx, rx_id) == (f["rx0"], f["rx1"])]
    #             tdoa_values_reverse = [-f["tdoa"] for f in self.tdoa_group.tdoas 
    #                                 if (ref_rx, rx_id) == (f["rx1"], f["rx0"])]
    #             if tdoa_values_direct:
    #                 tdoa_mat[i][j] = tdoa_values_direct[0] * 1e9  # Store the first value of the list
    #             if tdoa_values_reverse:
    #                 tdoa_mat[i][j] += tdoa_values_reverse[0] * 1e9  # Add the negative value
    #     return tdoa_mat


    # def get_subgroup(self, ref_rx):
    #     tdoa_data_used = [f for f in self.tdoa_group.tdoas if ref_rx in (f["rx0"], f["rx1"])]
    #     rx_used = np.unique([f["rx0"] for f in tdoa_data_used] + [f["rx1"] for f in tdoa_data_used])
    #     rx_pos_used = {key: self.pos_rx[key] for key in rx_used if key in self.pos_rx}
    #     tau = [f["tdoa"] for f in tdoa_data_used]
    #     return rx_pos_used, tau


    def solve_w_ref_receiver(self, ref_rx):
        rx_index = np.where(self.receiver_list == ref_rx)[0][0]
        tau = np.delete(self.tdoa_matrix[rx_index, :], rx_index)
        src, JDBest_min = AML_solver.tracking_solver_aml_2d(self.pos_rx_used, tau/1e9, ref_rx) 
        # rx_pos_used, tau = self.get_subgroup(ref_rx)
        # src, JDBest_min = AML_solver.tracking_solver_aml_2d(rx_pos_used, tau, ref_rx) 
        self.solved_loc = src
        return src, tau, JDBest_min


    def solve_w_first_receiver(self):
        src, JDBest_min = self.solve_w_ref_receiver(self.receiver_list[0])
        return src, JDBest_min


    def solve_w_best_receiver(self):
        checkTDOA = {}
        solved_locs = {}
        # Select the best receiver based on some criteria
        for _, ref_rx in enumerate(self.receiver_list):
            # rx_pos_used, tau = self.get_subgroup(ref_rx)
            src, tau, _ = self.solve_w_ref_receiver(ref_rx)

            T_trans = {f: np.linalg.norm(src[:2]-self.pos_rx_used[f][:2])/speed_of_light for f in self.pos_rx_used}
            refT = T_trans[ref_rx]
            calTDOA = [T_trans[f]-refT for f in T_trans if f!=ref_rx]

            checkTDOA[ref_rx] = np.linalg.norm(np.array(calTDOA)*1e9+np.array(tau))
            solved_locs[ref_rx] = src

        min_ref_rx = min(checkTDOA, key=checkTDOA.get)
        return solved_locs[min_ref_rx]
        # return solved_locs
    
    def output_median_location(self):
        # Output the median of solved locations
        pass