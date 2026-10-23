/*
Copyright 2025 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __YAML_LITE_H__
#define __YAML_LITE_H__

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace cv
{

/** @brief A node of the YAML subset that is used by the calibration files.
The supported constructs are block mappings, flow mappings, flow sequences,
scalars and the "!!opencv-matrix" nodes.
*/
class YamlNode
{
public:
	enum Type
	{
		NONE,
		SCALAR,
		MAP,
		SEQUENCE
	};

	Type type = NONE;
	std::string scalar;                          //!< value of a scalar node
	std::vector<std::pair<std::string, YamlNode>> map;    //!< entries of a mapping
	std::vector<YamlNode> sequence;              //!< elements of a sequence

	bool empty() const { return type == NONE; }
	bool isMap() const { return type == MAP; }
	bool isScalar() const { return type == SCALAR; }
	bool isSequence() const { return type == SEQUENCE; }

	/** @brief Returns the child node of a mapping or nullptr.
	@param key Name of the entry.
	@return The child node or nullptr if there is no such entry.
	*/
	const YamlNode* find(const std::string& key) const;

	/** @brief Returns the value of a scalar entry.
	@param key Name of the entry.
	@param defaultValue Value returned if the entry does not exist.
	@return The value of the entry.
	*/
	std::string getString(const std::string& key, const std::string& defaultValue = std::string()) const;

	/** @brief Returns the value of a scalar entry as a number.
	@param key Name of the entry.
	@param defaultValue Value returned if the entry does not exist.
	@return The value of the entry.
	*/
	double getDouble(const std::string& key, double defaultValue = 0.0) const;

	/** @brief Returns the value of a scalar entry as an integer.
	@param key Name of the entry.
	@param defaultValue Value returned if the entry does not exist.
	@return The value of the entry.
	*/
	int getInt(const std::string& key, int defaultValue = 0) const;

	/** @brief Converts the node into a matrix.
	The node has to hold the "rows", "cols", "dt" and "data" entries of an
	"!!opencv-matrix" node.
	@return The matrix or an empty matrix if the node does not hold one.
	*/
	Mat toMatrix() const;

	/** @brief Converts the entry of a mapping into a matrix.
	@param key Name of the entry.
	@return The matrix or an empty matrix if the entry does not hold one.
	*/
	Mat getMatrix(const std::string& key) const;
};

/** @brief Parses a YAML or JSON file.
The OpenCV file storage cannot read the flow mappings that hold a matrix,
which is the layout written by the common rig calibration tools, so the files
are parsed here instead.
@param filename Path of the file.
@return The root node of the document.
*/
YamlNode parseYamlFile(const String& filename);

} // namespace cv

#endif // !__YAML_LITE_H__
