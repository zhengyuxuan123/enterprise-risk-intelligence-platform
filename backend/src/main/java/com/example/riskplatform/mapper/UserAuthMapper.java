package com.example.riskplatform.mapper;
import com.example.riskplatform.entity.SysUser;import org.apache.ibatis.annotations.*;import java.util.List;
public interface UserAuthMapper{
 @Select("SELECT * FROM sys_user WHERE username=#{username} LIMIT 1")SysUser findByUsername(String username);
 @Select("SELECT DISTINCT p.permission_code FROM sys_permission p JOIN sys_role_permission rp ON rp.permission_id=p.id JOIN sys_user_role ur ON ur.role_id=rp.role_id WHERE ur.user_id=#{userId}")List<String> findPermissions(Long userId);
 @Select("SELECT DISTINCT r.role_code FROM sys_role r JOIN sys_user_role ur ON ur.role_id=r.id WHERE ur.user_id=#{userId}")List<String> findRoleCodes(Long userId);
 @Select("SELECT r.data_scope FROM sys_role r JOIN sys_user_role ur ON ur.role_id=r.id WHERE ur.user_id=#{userId} ORDER BY FIELD(r.data_scope,'ALL','ALL_READONLY','DEPT_AND_CHILD','DEPT','SELF') LIMIT 1")String findDataScope(Long userId);
 @Delete("DELETE FROM sys_user_role WHERE user_id=#{userId}")int deleteUserRoles(Long userId);
 @Insert("INSERT INTO sys_user_role(user_id,role_id) VALUES(#{userId},#{roleId})")int insertUserRole(@Param("userId")Long userId,@Param("roleId")Long roleId);
}
