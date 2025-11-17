import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List, Tuple


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_cvat_xml_collect_labels(xml_path: Path) -> Tuple[ET.ElementTree, List[str]]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    labels = set()
    for box in root.findall(".//box"):
        labels.add(box.attrib["label"])
    for poly in root.findall(".//polygon"):
        labels.add(poly.attrib["label"])
    return tree, sorted(labels)


def save_cvat_xml(root_elem: ET.Element, output_path: Path):
    """
    保存 CVAT XML，自动添加完整的 meta/labels 结构

    如果输入的 root_elem 已经包含完整结构，则直接保存
    否则自动提取标签并添加完整的 CVAT XML 结构
    """
    # 检查是否已经有完整的 CVAT 结构
    has_meta = root_elem.find('.//meta') is not None
    has_labels = root_elem.find('.//labels') is not None

    if has_meta and has_labels:
        # 已经有完整结构，直接保存
        tree = ET.ElementTree(root_elem)
        # 美化 XML
        xml_str = minidom.parseString(ET.tostring(root_elem)).toprettyxml(indent="  ")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        return

    # 否则，需要构建完整的 CVAT XML 结构
    # 提取所有唯一的标签
    unique_labels = set()
    for image in root_elem.findall('image'):
        for box in image.findall('box'):
            label = box.get('label')
            if label:
                unique_labels.add(label)
        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            if label:
                unique_labels.add(label)

    # 创建新的根元素
    new_root = ET.Element('annotations')

    # 添加 version
    version = ET.SubElement(new_root, 'version')
    version.text = '1.1'

    # 添加 meta 结构
    meta = ET.SubElement(new_root, 'meta')
    task = ET.SubElement(meta, 'task')

    task_id = ET.SubElement(task, 'id')
    task_id.text = '0'

    task_name = ET.SubElement(task, 'name')
    task_name.text = 'auto_labeled'

    task_size = ET.SubElement(task, 'size')
    task_size.text = str(len(root_elem.findall('image')))

    # 添加 labels 定义
    labels_elem = ET.SubElement(task, 'labels')

    for i, label_name in enumerate(sorted(unique_labels)):
        label = ET.SubElement(labels_elem, 'label')

        name = ET.SubElement(label, 'name')
        name.text = label_name

        color = ET.SubElement(label, 'color')
        # 生成不同的颜色
        color.text = f'#{(i * 50) % 256:02x}{(i * 100) % 256:02x}{(i * 150) % 256:02x}'

        label_type = ET.SubElement(label, 'type')
        label_type.text = 'any'

        attributes = ET.SubElement(label, 'attributes')

    # 复制所有 image 元素
    for image in root_elem.findall('image'):
        new_root.append(image)

    # 美化并保存
    xml_str = minidom.parseString(ET.tostring(new_root)).toprettyxml(indent="  ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)


def validate_cvat_xml(xml_path: Path) -> Tuple[bool, str]:
    """
    验证 CVAT XML 的完整性

    Returns:
        (is_valid, message)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 检查基本结构
        if root.tag != 'annotations':
            return False, "Root element should be 'annotations'"

        # 检查是否有 labels 定义
        labels = root.findall('.//labels/label')
        if not labels:
            return False, "No labels defined in <meta><task><labels>"

        # 检查是否有图片
        images = root.findall('image')
        if not images:
            return False, "No images found"

        # 提取标签名称
        label_names = set()
        for label in labels:
            name_elem = label.find('name')
            if name_elem is not None and name_elem.text:
                label_names.add(name_elem.text)

        # 检查 box 中的标签是否都在定义中
        undefined_labels = set()
        for box in root.findall('.//box'):
            box_label = box.get('label')
            if box_label and box_label not in label_names:
                undefined_labels.add(box_label)

        if undefined_labels:
            return False, f"Found undefined labels in boxes: {undefined_labels}"

        return True, f"Valid CVAT XML: {len(labels)} labels, {len(images)} images"

    except Exception as e:
        return False, f"Error parsing XML: {e}"