/***************************************************************************
 * 
 * Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file plugin_create.h
 * @author huangben(huangben@baidu.com)
 * @date 2018/10/23 14:16:18
 * @version $Revision$ 
 * @brief 
 *  
 **/
#pragma once

#include <unordered_map>
#include <string>

namespace baidu {
namespace mirana {
namespace poros {

class IPlugin {
public:
    virtual ~IPlugin() {}
    virtual const std::string who_am_i() = 0;
};

typedef IPlugin* (*plugin_creator_t)();
typedef std::unordered_map<std::string, plugin_creator_t> plugin_creator_map_t;

IPlugin* create_plugin(const std::string& plugin_name);
IPlugin* create_plugin(const std::string& plugin_name, const plugin_creator_map_t& plugin_creator_map);

void create_all_plugins(const plugin_creator_map_t& plugin_creator_map, 
        std::unordered_map<std::string, IPlugin*>& plugin_m);
//void create_all_plugins(std::unordered_map<std::string, IPlugin*>& plugin_m);

template <typename PluginType>
IPlugin* default_plugin_creator() {
    return new (std::nothrow)PluginType;
}

void register_plugin_creator(const std::string& plugin_name, plugin_creator_t creator);
void register_plugin_creator(const std::string& plugin_name, 
        plugin_creator_t creator, plugin_creator_map_t& plugin_creator_map);

template <typename PluginType>
void register_plugin_class(const std::string& plugin_name) {
    return register_plugin_creator(plugin_name, default_plugin_creator<PluginType>);
}

//推荐使用此版本
template <typename PluginType>
void register_plugin_class(const std::string& plugin_name, plugin_creator_map_t& plugin_creator_map) {
    return register_plugin_creator(plugin_name, default_plugin_creator<PluginType>, plugin_creator_map);
}

}//poros
}//mirana
}//baidu


/* vim: set ts=4 sw=4 sts=4 tw=100 */
