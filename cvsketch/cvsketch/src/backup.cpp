//
//        std::vector<std::pair<cv::Vec3b, int>> copyQuery(sorted_q_colorpalette);
//
//		const int area = double(max_width * max_height);
//		for (int iQColor = 0; iQColor < copyQuery.size(); iQColor++)
//		{
//			std::pair<cv::Vec3b, int> qc = copyQuery[iQColor];
//			int minIdx = 0;
//			double min_dist = std::numeric_limits<double>::max();
//			int qp = sorted_q_colorpalette[iQColor].second;
//			for (int iDBColor = 0; iDBColor < sorted_colorpalette.size(); iDBColor++)
//			{
//
//				std::pair<cv::Vec3b, int> dp = copyQuery[iDBColor];
//
//				double l = (qc.first[0] - dp.first[0]);
//				double a = (qc.first[1] - dp.first[1]);
//				double b = (qc.first[2] - dp.first[2]);
//				double tmp_dist = (std::pow(l, 2) + std::pow(a, 2) + std::pow(b, 2));
//				tmp_dist = std::sqrt(tmp_dist);
//
//				if(tmp_dist < min_dist)
//				{
//				    min_dist = tmp_dist;
//					minIdx = iDBColor;
//				}
//			}
//
//			int dbp = sorted_colorpalette[minIdx].second;
//			int diff;
//			std::pair<cv::Vec3b, int> rest_t;
//
//            if(copyQuery[minIdx].second != 0 || sorted_colorpalette[iQColor].second != 0)
//            {
//                if(qp > dbp)
//                {
//                    diff = qp - dbp;
//                    copyQuery[iQColor].second = diff;
//                    rest_t = std::make_pair(copyQuery[iQColor].first, (qp - diff));
//                    copyQuery.push_back(rest_t);
//                    q_stack.push_back(rest_t);
//                    db_stack.push_back(sorted_colorpalette[minIdx]);
//
//                }
//                else if(qp < dbp)
//                {
//                    diff = dbp - qp;
//                    copyQuery[minIdx].second = diff;
//                    rest_t = std::make_pair(sorted_colorpalette[minIdx].first, (dbp - diff));
//                    db_stack.push_back(rest_t);
//                    q_stack.push_back(copyQuery[iQColor]);
//                    sorted_colorpalette.push_back(rest_t);
//                    sorted_colorpalette[minIdx].second = 0;
//
//                }else
//                {
//                    q_stack.push_back(copyQuery[iQColor]);
//                    db_stack.push_back(sorted_colorpalette[minIdx]);
//                    copyQuery[minIdx].second = 0;
//                    sorted_colorpalette[iQColor].second = 0;
//                }
//            }
//
//		}
//
//        std::vector<std::pair<cv::Vec3b, int>> q_stack_sorted(q_stack);
//        std::vector<std::pair<cv::Vec3b, int>> db_stack_sorted(db_stack);
//        //vbs::Segmentation::sortPaletteByArea(q_stack, q_stack_sorted);
//        //vbs::Segmentation::sortPaletteByArea(db_stack, db_stack_sorted);
//
//        std::vector<std::pair<cv::Vec3b, float>> q_stack2(q_stack.size());
//        std::vector<std::pair<cv::Vec3b, float>> db_stack2(db_stack.size());
//
//
//
//
//        for (int i = 0; i < q_stack_sorted.size(); i++)
//        {
//            float weight = float(q_stack_sorted[i].second) / float(max_height * max_width);
//            q_stack2[i].first = q_stack_sorted[i].first;
//            q_stack2[i].second = weight;
//            std::cout << "Color: " << q_stack2[i].first << " \t - Area: " << float(q_stack2[i].second) << "%" << std::endl;
//        }
//
//        for (int i = 0; i < db_stack_sorted.size(); i++)
//        {
//            float weight = float(db_stack_sorted[i].second) / float(max_height * max_width);
//            db_stack2[i].first = db_stack_sorted[i].first;
//            db_stack2[i].second = weight;
//            std::cout << "Color: " << db_stack2[i].first << " \t - Area: " << float(db_stack2[i].second) << "%" << std::endl;
//        }
//
//
//
//        print_stack(q_stack2, stack_q_r);
//
//        cv::namedWindow(winNameAfterQueryStack);
//        show_image(stack_q_r, winNameAfterQueryStack, 0, stack_q_r.rows + 50);
//
//        print_stack(db_stack2, stack_db_r);
//        cv::namedWindow(winNameAfterDBStack);
//        show_image(stack_db_r, winNameAfterDBStack, stack_q_r.cols, stack_q_r.rows + 50 );
//        cv::waitKey(0);
////        cv::destroyWindow(winNameAfterQueryStack);
////        cv::destroyWindow(winNameAfterDBStack);
//        stack_q_r.release();
//        stack_db_r.release();
//
//        double dist = 0.0;
//		for(int i = 0; i < q_stack.size(); i++)
//		{
//			std::pair<cv::Vec3b, int> qc = q_stack[i];
//
//            int per1 = (qc.second / double(area)) * 100;
//
//				std::pair<cv::Vec3b, int> dbc = db_stack[i];
//
//                std::cout << "QC: " << qc.first << " DB: " << dbc.first << std::endl;
//
//				double l = (dbc.first[0] - qc.first[0]);
//				double a = (dbc.first[1] - qc.first[1]);
//		        double b = (dbc.first[2] - qc.first[2]);
//				double tmp_dist = (std::pow(l, 2) + std::pow(a, 2) + std::pow(b, 2));
//				tmp_dist = std::sqrt(tmp_dist);
//
//                std::cout << "Dist: " << tmp_dist << std::endl;
//
//                int per2 = (dbc.second / double(area)) * 100;
//				if (per1 != per2)
//				{
//					std::cout << "Per not the same" << std::endl;
//				}
//                dist += (tmp_dist * per1);
//
//
//
//			//dist = dist / double(sorted_colorpalette.size());
//		}
//
//		dist = dist / double(q_stack.size());
