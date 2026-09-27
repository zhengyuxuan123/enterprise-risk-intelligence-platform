package com.example.riskplatform.mapper;import org.apache.ibatis.annotations.Select;import java.util.List;
public interface CompanyScopeMapper{
 @Select("SELECT id FROM company WHERE owner_user_id=#{userId} AND status=1")List<Long> selectIdsByOwner(Long userId);
 @Select("SELECT id FROM company WHERE dept_id=#{deptId} AND status=1")List<Long> selectIdsByDept(Long deptId);
 @Select("WITH RECURSIVE dept_tree AS (SELECT id FROM sys_department WHERE id=#{deptId} UNION ALL SELECT d.id FROM sys_department d JOIN dept_tree dt ON d.parent_id=dt.id) SELECT id FROM company WHERE dept_id IN (SELECT id FROM dept_tree) AND status=1")List<Long> selectIdsByDeptTree(Long deptId);
}
