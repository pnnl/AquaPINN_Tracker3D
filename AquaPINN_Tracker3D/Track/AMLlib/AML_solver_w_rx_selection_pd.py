import numpy as np
from . import AML_solver
#
#speed_of_light = 299792458  # meters per second

class TDOAGroup:
    def __init__(self, tdoa_group, pos_rx, sound_speed=1500, noises=None, RSR=None):        
        self.tdoa_group = tdoa_group
        self.pos_rx     = pos_rx
        self.solved_loc = None  # Initialize solved_loc attribute
        self.receiver_list = self.get_receiver_list() 
        self.pos_rx_used = {key: self.pos_rx[key] for key in self.receiver_list if key in self.pos_rx}
        self.noises = noises
        self.sound_speed = sound_speed
        self.RSR = RSR
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
        # Extract unique receivers from both 'rx0' and 'rx1' columns
        receivers_rx0 = self.tdoa_group['rx0'].unique()
        receivers_rx1 = self.tdoa_group['rx1'].unique()
        # Combine and find unique receivers across both columns
        unique_receivers = np.unique(np.concatenate((receivers_rx0, receivers_rx1)))
        return unique_receivers


    ## more efficient code to get tdoa matrix
    def get_tdoa_matrix(self, tdoa_sync_method):
        num_receivers = len(self.receiver_list)
        tdoa_dict = {}

        # Populate the dictionary with TDOA values for each pair of receivers
        for index, row in self.tdoa_group.iterrows():
            rx0, rx1, tdoa = row['rx0'], row['rx1'], row[tdoa_sync_method]
            # For direct pair (rx0, rx1)
            if (rx0, rx1) in tdoa_dict:
                tdoa_dict[(rx0, rx1)].append(tdoa)
            else:
                tdoa_dict[(rx0, rx1)] = [tdoa]
            # For reverse pair (rx1, rx0), appending negative tdoa to indicate reverse direction
            if (rx1, rx0) in tdoa_dict:
                tdoa_dict[(rx1, rx0)].append(-tdoa)
            else:
                tdoa_dict[(rx1, rx0)] = [-tdoa]  # This initializes with -tdoa if not already present


        # Create the TDOA matrix using the dictionary
        tdoa_mat = np.zeros((num_receivers, num_receivers))
        for i, ref_rx in enumerate(self.receiver_list):
            for j, rx_id in enumerate(self.receiver_list):
                tdoa_values_forward = tdoa_dict.get((ref_rx, rx_id), [])
                tdoa_values_reverse = tdoa_dict.get((rx_id, ref_rx), [])
                tdoa_values = tdoa_values_forward + tdoa_values_reverse
                if tdoa_values:
                    tdoa_mat[i][j] = tdoa_values[0] # Store the first value of the list

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


    def solve_w_ref_receiver_internal(self, ref_rx, tdoa_sync_method, solved_dimension):
        self.tdoa_matrix   = self.get_tdoa_matrix(tdoa_sync_method)
        rx_index = np.where(self.receiver_list == ref_rx)[0][0]
        tau = np.delete(self.tdoa_matrix[rx_index, :], rx_index)
        # noises = self.noises
        src, JDBest_min = AML_solver.tracking_solver_aml(self.pos_rx_used, tau, ref_rx , 
                                                         solved_dimension=solved_dimension,
                                                         sound_speed=self.sound_speed,
                                                         noises=self.noises,
                                                         RSR=self.RSR) 
        # rx_pos_used, tau = self.get_subgroup(ref_rx)
        # src, JDBest_min = AML_solver.tracking_solver_aml_2d(rx_pos_used, tau, ref_rx) 
        self.solved_loc = src
        return src, tau, JDBest_min

    def solve_w_ref_receiver(self, ref_rx, tdoa_sync_method='tdoa'):
        self.tdoa_matrix   = self.get_tdoa_matrix(tdoa_sync_method)
        rx_index = np.where(self.receiver_list == ref_rx)[0][0]
        tau = np.delete(self.tdoa_matrix[rx_index, :], rx_index)
        src, JDBest_min = AML_solver.tracking_solver_aml(self.pos_rx_used, tau, ref_rx,
                                                         sound_speed=self.sound_speed,
                                                         RSR=self.RSR) 
        # rx_pos_used, tau = self.get_subgroup(ref_rx)
        # src, JDBest_min = AML_solver.tracking_solver_aml_2d(rx_pos_used, tau, ref_rx) 
        return src


    def solve_w_first_receiver(self):
        src, JDBest_min = self.solve_w_ref_receiver(self.receiver_list[0])
        return src, JDBest_min


    def solve_w_best_receiver(self, tdoa_sync_method='tdoa', solved_dimension=2):
        checkTDOA = {}
        solved_locs = {}
        # Select the best receiver based on some criteria
        for _, ref_rx in enumerate(self.receiver_list):
            # rx_pos_used, tau = self.get_subgroup(ref_rx)
            src, tau, _ = self.solve_w_ref_receiver_internal(ref_rx, tdoa_sync_method, solved_dimension)

            T_trans = {f: np.linalg.norm(src[:solved_dimension]-self.pos_rx_used[f][:solved_dimension])/self.sound_speed for f in self.pos_rx_used}
            refT = T_trans[ref_rx]
            calTDOA = [T_trans[f]-refT for f in T_trans if f!=ref_rx]

            checkTDOA[ref_rx] = np.linalg.norm(np.array(calTDOA)+np.array(tau))
            solved_locs[ref_rx] = src

        min_ref_rx = min(checkTDOA, key=checkTDOA.get)

        # if 0 in solved_locs.keys():
        #     min_ref_rx = 0        
        # if 1 in solved_locs.keys():
        #     min_ref_rx = 1
        return solved_locs[min_ref_rx], min_ref_rx
        # return solved_locs
    
    def output_median_location(self):
        # Output the median of solved locations
        pass