/***************************************************************************
 * 
 * Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file plugin_create.cpp
 * @author huangben(huangben@baidu.com)
 * @date 2018/10/23 14:16:18
 * @version $Revision$ 
 * @brief 
 *  
 **/
#include "poros/iplugin/plugin_create.h"
#include "poros/log/poros_logging.h"

namespace baidu {
namespace mirana {
namespace poros {

    static plugin_creator_map_t _g_creator_map;

    void register_plugin_creator(const std::string& plugin_name, plugin_creator_t creator) {
        if (_g_creator_map.find(plugin_name) != _g_creator_map.end()) {
            //throw bsl::KeyAlreadyExistException() << BSL_EARG
            //    << "[plugin_name:" << plugin_name << "]";
            LOG(ERROR) << plugin_name << " had resiterd! there have more than 1 plugin use samename";
        }
        _g_creator_map[plugin_name] = creator;
    }
    
    void register_plugin_creator(const std::string& plugin_name, plugin_creator_t creator, 
            plugin_creator_map_t& plugin_creator_map) {

        if (plugin_creator_map.find(plugin_name) != plugin_creator_map.end()) {
            //throw bsl::KeyAlreadyExistException() << BSL_EARG
            //    << "[plugin_name:" << plugin_name << "]";
            LOG(ERROR) << plugin_name << " had resiterd! there have more than 1 plugin use samename";
        }
        plugin_creator_map[plugin_name] = creator;
    }

    IPlugin* create_plugin(const std::string& plugin_name) {
        plugin_creator_map_t::const_iterator it;
        
        it = _g_creator_map.find(plugin_name);
        if (it == _g_creator_map.end()) {
            LOG(FATAL) << "No such plugin type:" << plugin_name;
            return NULL;
        }
        return it->second();
    }
    
    IPlugin* create_plugin(const std::string& plugin_name, const plugin_creator_map_t& plugin_creator_map) {
        plugin_creator_map_t::const_iterator it;
        
        it = plugin_creator_map.find(plugin_name);
        if (it == plugin_creator_map.end()) {
            LOG(FATAL) << "No such plugin type:" << plugin_name;
            return NULL;
        }
        return it->second();
    }

    //void create_all_plugins(std::unordered_map<std::string, IPlugin*>& plugin_m) {
    //    for (auto& e : _g_creator_map) {
    //        plugin_m[e.first] = e.second();
    //    }
    //}
    void create_all_plugins(const plugin_creator_map_t& plugin_creator_map, 
            std::unordered_map<std::string, IPlugin*>& plugin_m) {
        for (auto& e : plugin_creator_map) {
            plugin_m[e.first] = e.second();
        }
    }
    
}//poros
}//mirana
}//baidu


/* vim: set ts=4 sw=4 sts=4 tw=100 */
